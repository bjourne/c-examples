// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.
// This kernel computes C = A * B, where
//     A is a N x K matrix
//     B is a K x M matrix
//     C is a N x M matrix
#include "programs/opencl/matmul_fpga_config.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

////////////////////////////////////////////////////////////////////////
// Macro utility
////////////////////////////////////////////////////////////////////////
#define POW2_REM(i, v)              ((i) & ((v) - 1))

#define DEBUG 0
#if DEBUG==1
#define ASSERT(cond)                if (!(cond)) { printf("Cond: %s failed!\n", #cond); }
#else
#define ASSERT(cond)
#endif

#define FPGA_REG1(x)                __fpga_reg((x))
#define FPGA_REG2(x)                __fpga_reg(__fpga_reg((x)))

#define VECTOR_FLOAT1_ZERO          0.0f
#define VECTOR_FLOAT2_ZERO          (float2)(0.0f, 0.0f)
#define VECTOR_FLOAT4_ZERO          (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO          (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO         (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)

// Shift register size
#define SHIFT_REG_SIZE              (PE_S * PE_S)

// One store per row in the systolic array
#define SHIFT_REGS_PER_Y            (PE_S * PE_S * PE_S)

// Number of messages per block of A
#define A_BLOCK_N_MSGS      (PE_S * PE_S * X_SCALE)
#define N_B_LOADS           (PE_S * PE_S * X_SCALE)

////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////
#if VECTOR_SIZE==1
#define VECTOR_ZERO         VECTOR_FLOAT1_ZERO
typedef float vfloat;
#elif VECTOR_SIZE==2
typedef float2 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT2_ZERO
#elif VECTOR_SIZE==4
typedef float4 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT4_ZERO
#elif VECTOR_SIZE==8
typedef float8 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT8_ZERO
#elif VECTOR_SIZE==16
typedef float16 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT16_ZERO
#else
#error Unsupported VECTOR_SIZE
#endif

////////////////////////////////////////////////////////////////////////
// Structs
////////////////////////////////////////////////////////////////////////
typedef struct {
    float data[PE_S];
} cols_floats;

typedef struct {
    vfloat data;
    // indicates a new row/column pair
    bool c;
} vfloat_bool;

////////////////////////////////////////////////////////////////////////
// Channels
////////////////////////////////////////////////////////////////////////
#define CHAN_DEPTH      256

channel vfloat_bool ch_load_a __attribute__((depth(CHAN_DEPTH)));
channel vfloat ch_load_b __attribute__((depth(CHAN_DEPTH)));
channel cols_floats ch_store_c __attribute__((depth(CHAN_DEPTH)));

// We send every row of A tiles K times. Then a zero row to drain the
// array.
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
loadA(global vfloat* restrict A, uint M, uint N, uint K) {
    for (uint n = 0; n < N; n++) {
        for (uint k = 0; k < K; k++) {
            for (uint m = 0; m < M; m++) {
                vfloat_bool buf;
                // Only true for the second block
                buf.c = m == 1;
                for (uint i = 0; i < A_BLOCK_N_MSGS; i++) {
                    buf.data = A[(M*n + m) * A_BLOCK_N_MSGS + i];
                    write_channel_intel(ch_load_a, buf);
                }
            }
        };
    }

    vfloat_bool buf;
    buf.data = VECTOR_ZERO;
    for (uint i = 0; i < M; i++) {
        buf.c = i == 1;
        for (uint j = 0; j < A_BLOCK_N_MSGS; j++) {
            write_channel_intel(ch_load_a, buf);
        }
    }
}

// Send the whole B matrix N times column by column. Then a zero
// column to drain the array.
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
loadB(global vfloat* restrict B, uint M, uint N, uint K) {

    uint n_msgs_per_col = M * X_SCALE * PE_S * PE_S;
    uint n_msgs_tot = n_msgs_per_col * K;

    vfloat buf;
    for (uint n = 0; n < N; n++) {
        for (uint i = 0; i < n_msgs_tot; i++) {
            buf = B[i];
            write_channel_intel(ch_load_b, buf);
        }
    }

    buf = VECTOR_ZERO;
    for (uint i = 0; i < n_msgs_per_col; i++) {
        write_channel_intel(ch_load_b, buf);
    }
}

float
PE(bool clear, vfloat valA, vfloat valB, float *acc) {
    float oldAcc = FPGA_REG1(acc[0]);
    float sum = clear ? 0.0f : oldAcc;

#if VECTOR_SIZE==1
    sum += valA * valB;
#else
#pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
        sum += valA[i] * valB[i];
    }
#endif

#pragma unroll
    for (int i = 0; i < SHIFT_REG_SIZE - 1; i++) {
        acc[i] = acc[i + 1];
    }
    acc[SHIFT_REG_SIZE - 1] = sum;
    return oldAcc;
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
void
kernel monolithic() {
    // internal feeder A and B storage, banked, 1 bank per feeder
    vfloat __attribute__((memory,
                          numbanks(PE_S),
                          bankwidth(sizeof(vfloat)),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_a[2][PE_S][X_SCALE][PE_S];
    vfloat __attribute__((memory,
                          numbanks(PE_S),
                          bankwidth(sizeof(vfloat)),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_b[2][PE_S][X_SCALE][PE_S];


    // internal PE storage for accumulations, PE_S * PE_S shift registers
    float acc[PE_S][PE_S][SHIFT_REG_SIZE];
    // shift register for drain, one per column
    float drain[PE_S][SHIFT_REG_SIZE * (PE_S - 1) + 1];

    uint storecount = SHIFT_REGS_PER_Y;
    bool new_c_tile = false;

    while (1) {
        for (uint side = 0; side < 2; side++) {
            for (uint counter = 0; counter < PE_S * PE_S * X_SCALE; counter++) {

                vfloat_bool valA = read_channel_intel(ch_load_a);

                // save latest row_col_pair
                if ((!new_c_tile && valA.c) & 1) {
                    storecount = 0;
                }
                new_c_tile = valA.c;

                vfloat valB = read_channel_intel(ch_load_b);

                vfloat_bool fedA[PE_S];
                vfloat fedB[PE_S];

                // Rename counter to break dependency to the loop
                // index.
                uint counter2 = counter;
#pragma unroll
                for (uint e = 0; e < PE_S; e++) {
                    if (counter2 / (PE_S * X_SCALE) == e) {
                        uchar w_vec = POW2_REM(counter2, X_SCALE);
                        uchar w_addr = POW2_REM(counter2, PE_S * X_SCALE) / X_SCALE;
                        mem_a[side][w_addr][w_vec][e] = valA.data;
                        mem_b[side][w_addr][w_vec][e] = valB;
                    }

                    uchar r_vec = counter2 / (PE_S * PE_S);
                    uchar r_addr_a = POW2_REM(counter2, PE_S * PE_S) / PE_S;
                    uchar r_addr_b = POW2_REM(counter2, PE_S);

                    fedA[e].data = mem_a[!side][r_addr_a][r_vec][e];
                    fedA[e].c = (counter2 < (PE_S * PE_S)) & valA.c;
                    fedB[e] = mem_b[!side][r_addr_b][r_vec][e];
                }

#pragma unroll
                for (uint y = 0; y < PE_S; y++) {
#pragma unroll
                    for (uint x = 0; x < PE_S; x++) {
                        // compute and store outputs in shift register
                        float result = PE(fedA[y].c, fedA[y].data, fedB[x], acc[y][x]);
                        if (fedA[y].c) {
                            drain[x][y * SHIFT_REG_SIZE] = result;
                        }
                        fedA[y].data = FPGA_REG2(fedA[y].data);
                        fedA[y].c = FPGA_REG2(fedA[y].c);
                        fedB[x] = FPGA_REG2(fedB[x]);
                    }
                }

                cols_floats results;
#pragma unroll
                for (uint x = 0; x < PE_S; x++) {
                    results.data[x] = drain[x][0];
#pragma unroll
                    for (uint i = 0; i < SHIFT_REG_SIZE * (PE_S - 1); i++) {
                        drain[x][i] = drain[x][i + 1];
                    }
                }
                if (storecount < SHIFT_REGS_PER_Y) {
                    write_channel_intel(ch_store_c, results);
                }
                storecount++;
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
store(global float * restrict C, uint N, uint K) {

    // We read and discard this many messages
    for (uint i = 0; i < SHIFT_REGS_PER_Y; i++) {
        read_channel_intel(ch_store_c);
    }
    uint c_n_msgs = K * N * PE_S * PE_S * PE_S;
    for (uint i = 0; i < c_n_msgs; i++) {
        cols_floats d = read_channel_intel(ch_store_c);
#pragma unroll
        for (uint j = 0; j < PE_S; j++) {
            C[PE_S * i + j] = d.data[j];
        }
    }
}
