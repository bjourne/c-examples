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

#define VECTOR_LONG4_ZERO           (long4)(0, 0, 0, 0)
#define VECTOR_LONG8_ZERO           (long8)(VECTOR_LONG4_ZERO, VECTOR_LONG4_ZERO)

// Shift register size
#define SHIFT_REG_SIZE              (PE_S * PE_S)

// Number of messages per block of A, B, or C
#define N_AB_BLOCK_MSGS             (PE_S * PE_S * X_SCALE)
#define N_C_BLOCK_MSGS              (PE_S * PE_S * PE_S)

////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////
#if TYPE_SEL==1

typedef long type;

#if V_SIZE==8
#define VECTOR_ZERO         VECTOR_LONG8_ZERO
typedef long8 vtype;
#else
#error Unsupported V_SIZE
#endif

#elif TYPE_SEL==2

typedef float type;

#if V_SIZE==1
#define VECTOR_ZERO         VECTOR_FLOAT1_ZERO
typedef float vtype;
#elif V_SIZE==2
typedef float2 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT2_ZERO
#define VECTOR_FMT          "v2hlf"
#elif V_SIZE==4
typedef float4 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT4_ZERO
#elif V_SIZE==8
typedef float8 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT8_ZERO
#elif V_SIZE==16
typedef float16 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT16_ZERO
#else
#error Unsupported V_SIZE
#endif

#else

#error Unsupported TYPE_SEL

#endif

////////////////////////////////////////////////////////////////////////
// Structs
////////////////////////////////////////////////////////////////////////
typedef struct {
    type data[PE_S];
} cols_data;

typedef struct {
    vtype data;
    // indicates a new row/column pair
    bool c;
} vtype_bool;

////////////////////////////////////////////////////////////////////////
// Channels
////////////////////////////////////////////////////////////////////////
#define CHAN_DEPTH      256

channel vtype_bool ch_load_a __attribute__((depth(CHAN_DEPTH)));
channel vtype ch_load_b __attribute__((depth(CHAN_DEPTH)));
channel cols_data ch_store_c __attribute__((depth(CHAN_DEPTH)));

// We send every row of A tiles K times. Then a zero row to drain the
// array.
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
load_a(global const vtype* restrict A, uint N, uint M, uint K) {
    for (uint n = 0; n < N; n++) {
        for (uint k = 0; k < K; k++) {
            for (uint m = 0; m < M; m++) {
                vtype_bool buf;
                // Only true for the second block
                buf.c = m == 1;
                for (uint i = 0; i < N_AB_BLOCK_MSGS; i++) {
                    buf.data = A[(M * n + m) * N_AB_BLOCK_MSGS + i];
                    write_channel_intel(ch_load_a, buf);
                }
            }
        };
    }

    vtype_bool buf;
    buf.data = VECTOR_ZERO;
    for (uint m = 0; m < M; m++) {
        buf.c = m == 1;
        for (uint i = 0; i < N_AB_BLOCK_MSGS; i++) {
            write_channel_intel(ch_load_a, buf);
        }
    }
}

// Send the whole B matrix N times column by column. Then a zero
// column to drain the array.
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
load_b(global const vtype* restrict B, uint N, uint M, uint K) {
    for (uint n = 0; n < N; n++) {
        for (uint i = 0; i < K * M * N_AB_BLOCK_MSGS; i++) {
            write_channel_intel(ch_load_b, B[i]);
        }
    }
    for (uint i = 0; i < M * N_AB_BLOCK_MSGS; i++) {
        write_channel_intel(ch_load_b, VECTOR_ZERO);
    }
}

type
PE(bool clear, vtype valA, vtype valB, type *acc) {
    type oldAcc = FPGA_REG1(acc[0]);
    type sum = clear ? 0 : oldAcc;

#if V_SIZE==1
    sum += valA * valB;
#else
#pragma unroll
    for (int i = 0; i < V_SIZE; i++) {
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
    vtype __attribute__((memory,
                          numbanks(PE_S),
                          bankwidth(sizeof(vtype)),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_a[2][PE_S][X_SCALE][PE_S];
    vtype __attribute__((memory,
                          numbanks(PE_S),
                          bankwidth(sizeof(vtype)),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_b[2][PE_S][X_SCALE][PE_S];

    // internal PE storage for accumulations, PE_S * PE_S shift registers
    type acc[PE_S][PE_S][SHIFT_REG_SIZE];
    // shift register for drain, one per column
    type drain[PE_S][SHIFT_REG_SIZE * (PE_S - 1) + 1];

    uint storecount = N_C_BLOCK_MSGS;
    bool new_c_tile = false;

    while (1) {
        for (uint side = 0; side < 2; side++) {
            for (uint counter = 0; counter < N_AB_BLOCK_MSGS; counter++) {

                vtype_bool valA = read_channel_intel(ch_load_a);

                // save latest row_col_pair
                if ((!new_c_tile && valA.c) & 1) {
                    storecount = 0;
                }
                new_c_tile = valA.c;

                vtype valB = read_channel_intel(ch_load_b);

                vtype fedA[PE_S];
                vtype fedB[PE_S];

                // Rename counter to break dependency to the loop
                // index.
                uint counter2 = counter;
                bool clear = (counter2 < (PE_S * PE_S)) & valA.c;
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

                    fedA[e] = mem_a[!side][r_addr_a][r_vec][e];
                    fedB[e] = mem_b[!side][r_addr_b][r_vec][e];
                }

#pragma unroll
                for (uint y = 0; y < PE_S; y++) {
#pragma unroll
                    for (uint x = 0; x < PE_S; x++) {
                        // compute and store outputs in shift register
                        type result = PE(clear, fedA[y], fedB[x], acc[y][x]);
                        if (clear) {
                            drain[x][y * SHIFT_REG_SIZE] = result;
                        }
                        fedA[y] = FPGA_REG2(fedA[y]);
                        fedB[x] = FPGA_REG2(fedB[x]);
                    }
                    //Unclear whether this is needed.
                    clear = FPGA_REG2(clear);
                }

                cols_data results;
#pragma unroll
                for (uint x = 0; x < PE_S; x++) {
                    results.data[x] = drain[x][0];
#pragma unroll
                    for (uint i = 0; i < SHIFT_REG_SIZE * (PE_S - 1); i++) {
                        drain[x][i] = drain[x][i + 1];
                    }
                }
                if (storecount < N_C_BLOCK_MSGS) {
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
store(global type * restrict C, uint N, uint M, uint K) {

    // We read and discard this many messages
    for (uint i = 0; i < N_C_BLOCK_MSGS; i++) {
        read_channel_intel(ch_store_c);
    }
    uint c_n_msgs = K * N * N_C_BLOCK_MSGS;
    for (uint i = 0; i < c_n_msgs; i++) {
        cols_data d = read_channel_intel(ch_store_c);
#pragma unroll
        for (uint j = 0; j < PE_S; j++) {
            C[PE_S * i + j] = d.data[j];
        }
    }
}
