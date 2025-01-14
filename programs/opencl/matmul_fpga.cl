/* Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved. */
/* Permission is hereby granted, free of charge, to any person obtaining a copy of this */
/* software and associated documentation files (the "Software"), to deal in the Software */
/* without restriction, including without limitation the rights to use, copy, modify, merge, */
/* publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to */
/* whom the Software is furnished to do so, subject to the following conditions: */
/* The above copyright notice and this permission notice shall be included in all copies or */
/* substantial portions of the Software. */

/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES */
/* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND */
/* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT */
/* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, */
/* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING */
/* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR */
/* OTHER DEALINGS IN THE SOFTWARE. */

/* This agreement shall be governed in all respects by the laws of the State of California and */
/* by the laws of the United States of America. */
/* This kernel computes C = A * B, where */
/*     A is a N x K matrix */
/*     B is a K x M matrix */
/*     C is a N x M matrix */
#include "programs/opencl/matmul_fpga_config.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

////////////////////////////////////////////////////////////////////////
// Macro utility
////////////////////////////////////////////////////////////////////////
#define POW2_REM(i, v)              ((i) & ((v) - 1))

#define TRUNC(i, v)                 (((i) / (v)) * (v))

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
#define SHIFT_REG_SIZE              (PE_Y * PE_X)

// One store per row in the systolic array
#define SHIFT_REGS_PER_Y            (PE_Y * PE_X * PE_Y)

// Number of messages per block of A
#define A_BLOCK_N_MSGS      (PE_Y * PE_Y * X_SCALE)
#define N_B_LOADS           (PE_X * PE_X * X_SCALE)

#define SWAP_RANGE          (PE_Y * PE_X * X_SCALE)

// Try to load B as late as possible, so that if there is enough time
// and not enough DDR bandwidth, we can load all of A and then load
// all of B
#define FIRST_B_LOAD        (SWAP_RANGE - N_B_LOADS)

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
    float data[PE_X];
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

                    vfloat v = A[(M*n + m) * A_BLOCK_N_MSGS + i];
                    buf.data = v;
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

    uint n_msgs_per_col = M * X_SCALE * PE_X * PE_X;
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

// lo_counter: [0, SWAP_RANGE)
vfloat_bool
FeederA(vfloat_bool new,
        vfloat mem_a[2][PE_Y][X_SCALE][PE_Y],
        uint counter, uint y, uint side) {

    if (counter / (PE_Y * X_SCALE) == y) {
        uchar vector = POW2_REM(counter, X_SCALE);
        uchar col = POW2_REM(counter, PE_Y * X_SCALE) / X_SCALE;

        mem_a[side][col][TRUNC(vector, 1)][y] = new.data;
    }

    uchar col = POW2_REM(counter, SHIFT_REG_SIZE) / PE_X;
    uchar vector = counter / SHIFT_REG_SIZE;


    vfloat_bool val;
    val.data = mem_a[!side][col][TRUNC(vector, 1) ][y];
    val.c = (counter < SHIFT_REG_SIZE) & new.c;
    return val;
}


// counter is the global counter, which should align with FeederA's counter.
// load_counter is the counter for writing into the feeders,
// which may start at some point in the overall counter range (once FeederA is finished loading)
vfloat
FeederB(vfloat new,
        vfloat mem_b[2][PE_X][X_SCALE][PE_X],
        uint load_counter, uint col, uint counter, uint side) {

    bool do_write = load_counter / (PE_X * X_SCALE) == col;

    if (do_write) {
        uchar row = POW2_REM(load_counter, PE_X * X_SCALE) / X_SCALE;
        uchar vector = POW2_REM(load_counter, X_SCALE);

        mem_b[side][row][vector][col] = new;
    }

    uchar row = POW2_REM(counter, PE_X);
    uchar vector = counter / (PE_Y * PE_X);

    return mem_b[!side][row][TRUNC(vector, 1)][col];
}

float
PE(vfloat_bool valA, vfloat valB, float *acc) {
    float oldAcc = FPGA_REG1(acc[0]);
    float sum = valA.c ? 0.0f : oldAcc;

#if VECTOR_SIZE==1
    sum += valA.data * valB;
#else
#pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
        sum += valA.data[i] * valB[i];
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
                          numbanks(PE_Y),
                          bankwidth(sizeof(vfloat)),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_a[2][PE_Y][X_SCALE][PE_Y];
    vfloat __attribute__((memory,
                          numbanks(PE_X),
                          bankwidth(sizeof(vfloat)),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_b[2][PE_X][X_SCALE][PE_X];


    // internal PE storage for accumulations, PE_Y x PE_X shift registers
    float acc[PE_Y][PE_X][SHIFT_REG_SIZE];
    // shift register for drain, one per column
    float drain[PE_X][SHIFT_REG_SIZE * (PE_Y - 1) + 1];

    uint storecount = SHIFT_REGS_PER_Y;
    bool new_c_tile = false;

    while (1) {
        for (uint side = 0; side < 2; side++) {
            for (uint counter = 0; counter < SWAP_RANGE; counter++) {
                vfloat_bool valA;
                vfloat valB;

                if (counter < A_BLOCK_N_MSGS) {
                    valA = read_channel_intel(ch_load_a);

                    // save latest row_col_pair
                    if ((!new_c_tile && valA.c) & 1) {
                        storecount = 0;
                    }
                    new_c_tile = valA.c;
                }

                // Recover last known row_col_pair
                valA.c = new_c_tile;

                // Serialize the two reads to reduce burstiness.
                if (counter >= FIRST_B_LOAD) {
                    valB = read_channel_intel(ch_load_b);
                }

                // Feeders use privatized counters

                // Get feeder A data
                vfloat_bool fedA[PE_Y];
                uint counterA = counter;
#pragma unroll
                for (uint y = 0; y < PE_Y; y++) {
                    fedA[y] = FeederA(valA, mem_a, counterA, y, side);
                    valA.data = FPGA_REG2(valA.data);
                    valA.c = FPGA_REG2(valA.c);
                    counterA = FPGA_REG2(counterA);
                }

                // Get feeder B data
                vfloat fedB[PE_X];
                uint counterB = counter;
#pragma unroll
                for (int x = 0; x < PE_X; x++) {
                    // the indexing matches the serialization of the ch_load_b reads
                    fedB[x] = FeederB(valB, mem_b, counterB - FIRST_B_LOAD, x, counterB, side);
                    valB = FPGA_REG2(valB);
                    counterB = FPGA_REG2(counterB);
                }

#pragma unroll
                for (uint y = 0; y < PE_Y; y++) {
#pragma unroll
                    for (uint x = 0; x < PE_X; x++) {
                        // compute and store outputs in shift register
                        float result = PE(fedA[y], fedB[x], acc[y][x]);
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
                for (uint x = 0; x < PE_X; x++) {
                    results.data[x] = drain[x][0];
#pragma unroll
                    for (uint i = 0; i < SHIFT_REG_SIZE * (PE_Y - 1); i++) {
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
    uint c_n_msgs = K * N * PE_X * PE_Y * PE_Y;
    for (uint i = 0; i < c_n_msgs; i++) {
        cols_floats d = read_channel_intel(ch_store_c);
#pragma unroll
        for (uint j = 0; j < PE_X; j++) {
            C[PE_X * i + j] = d.data[j];
        }
    }
}
