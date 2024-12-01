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

// How many floats in each storage chunk. Must be n**2 < 256.
#define STORE_WIDTH 16

#define VECTOR_FLOAT4_ZERO          (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO          (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO         (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)


// The number of rows rounded up to the next power of 2
#if PE_Y <= 1
#define BANK_Y 1
#elif PE_Y <= 2
#define BANK_Y 2
#elif PE_Y <= 4
#define BANK_Y 4
#elif PE_Y <= 8
#define BANK_Y 8
#elif PE_Y <= 16
#define BANK_Y 16
#elif PE_Y <= 32
#define BANK_Y 32
#elif PE_Y <= 64
#define BANK_Y 64
#elif PE_Y <= 128
#define BANK_Y 128
#elif PE_Y <= 256
#define BANK_Y 256
#else
#error "PE_Y too large, BANK_Y cannot be defined"
#endif

// The number of columns rounded up to the next power of 2
#if PE_X <= 1
#define BANK_X 1
#elif PE_X <= 2
#define BANK_X 2
#elif PE_X <= 4
#define BANK_X 4
#elif PE_X <= 8
#define BANK_X 8
#elif PE_X <= 16
#define BANK_X 16
#elif PE_X <= 32
#define BANK_X 32
#elif PE_X <= 64
#define BANK_X 64
#elif PE_X <= 128
#define BANK_X 128
#elif PE_X <= 256
#define BANK_X 256
#else
#error "PE_X too large, BANK_X cannot be defined"
#endif

// Shift register size
#define SHIFT_REG_SIZE        (Y_INTERLEAVED * X_INTERLEAVED)

// Defines the width of channels in number of vectors.
#ifndef LVEC
#define LVEC 1
#endif


// Number of vectors per block of A
#define A_BLOCK_N_VECTORS           (A_BLOCK_Y * A_BLOCK_X / VECTOR_SIZE)

// Number of msgs pre block of A
#define A_BLOCK_N_MSGS              A_BLOCK_N_VECTORS / LVEC


// Vectors in an A block row.
#define X_VECS                    (A_BLOCK_X / VECTOR_SIZE)

#define SWAP_RANGE                  (Y_INTERLEAVED * X_INTERLEAVED * X_VECS)
#define RANGE                       (2 * SWAP_RANGE)

////////////////////////////////////////////////////////////////////////
// Sanity checking
////////////////////////////////////////////////////////////////////////
#if STORE_WIDTH < PE_X
#error "STORE_WIDTH must be >= PE_X!"
#endif


////////////////////////////////////////////////////////////////////////
// Macro utility
////////////////////////////////////////////////////////////////////////
#define POW2_REM(i, v)              ((i) & ((v) - 1))


////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////
#if VECTOR_SIZE==4
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
    vfloat data[LVEC];
} n_vfloat;

typedef struct {
    vfloat data[LVEC];
    // indicates a new row/column pair
    bool  c;
} n_vfloat_bool;

typedef struct {
    vfloat data;
    // indicates a new row/column pair
    bool  c;
} vfloat_bool;

////////////////////////////////////////////////////////////////////////
// Channels
////////////////////////////////////////////////////////////////////////
channel n_vfloat_bool ch_load_a __attribute__((depth(64)));
channel n_vfloat ch_load_b __attribute__((depth(64)));
channel cols_floats ch_store_c __attribute__((depth(64)));


__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
loadA(global volatile vfloat* restrict A, uint M, uchar N, uchar K) {
    for (uint n = 0; n < N; n++) {
        for (uint k = 0; k < K; k++) {
            for (uint m = 0; m < M; m++) {
                // Only true for the first block
                n_vfloat_bool send_buf;
                send_buf.c = m == 1;
                for (uint i = 0; i < A_BLOCK_N_MSGS; i++) {
#pragma unroll
                    for (int j = 0; j < LVEC; j++) {
                        send_buf.data[j] =
                            A[LVEC * ((M*n + m) * A_BLOCK_N_MSGS + i) + j];
                    }
                    write_channel_intel(ch_load_a, send_buf);
                }
            }
        };
    }

    // This code is weird
    n_vfloat_bool send_buf;
    #pragma unroll
    for (int i = 0; i < LVEC; i++) {
        send_buf.data[i] = VECTOR_ZERO;
    }
    for (uint i = 0; i < M; i++) {
        send_buf.c = i == 1;
        for (uint j = 0; j < A_BLOCK_N_MSGS; j++) {
            write_channel_intel(ch_load_a, send_buf);
        }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
loadB(global volatile vfloat* restrict B,
      uint b_n_vectors_per_col,
      uint b_n_vectors_tot,
      uchar N) {

    n_vfloat send_buf;

    for (uint times = 0; times < N; times++) {
        for (uint v_id = 0; v_id < b_n_vectors_tot / LVEC; v_id++) {
#pragma unroll
            for (int i = 0; i < LVEC; i++) {
                send_buf.data[i] = B[v_id * LVEC + i];
            }
            write_channel_intel(ch_load_b, send_buf);
        }
    }
    // done reload and forwarding the matrix data?
#pragma unroll
    for (int j = 0; j < LVEC; j++) {
        send_buf.data[j] = VECTOR_ZERO;
    }
    uint n_pkts_per_col = b_n_vectors_per_col / LVEC;
    for (uint i = 0; i < n_pkts_per_col; i++) {
        write_channel_intel(ch_load_b, send_buf);
    }
}

vfloat_bool
FeederA(n_vfloat_bool newVal,
        vfloat double_buffer[2][Y_INTERLEAVED][X_VECS][BANK_Y],
        uint counter, int row) {

    uint masked_counter = POW2_REM(counter, SWAP_RANGE);

    bool write_to_buffer =
        (masked_counter * LVEC / (Y_INTERLEAVED * X_VECS)) == row;
    bool new_row_col_pair =
        masked_counter < (Y_INTERLEAVED * X_INTERLEAVED);
    bool buffer_id_to_write_to = (counter / SWAP_RANGE) & 1;
    bool buffer_id_to_feed_to_sysarr = !buffer_id_to_write_to;

    if (write_to_buffer) {
        uchar buffer_vector_to_write_to = POW2_REM(counter * LVEC, X_VECS);
        uchar buffer_row_to_write_to = POW2_REM(counter * LVEC, Y_INTERLEAVED * X_VECS) / X_VECS;
        #pragma unroll
        for (int i = 0; i < LVEC; i++) {
            double_buffer[buffer_id_to_write_to][buffer_row_to_write_to][(buffer_vector_to_write_to / LVEC) * LVEC + i][row] = newVal.data[i];
        }
    }

    uchar buffer_row_to_feed_to_sysarr =
        POW2_REM(counter, Y_INTERLEAVED * X_INTERLEAVED) / X_INTERLEAVED;
    uchar buffer_vector_to_feed_to_sysarr =
        masked_counter / (Y_INTERLEAVED * X_INTERLEAVED);

    vfloat_bool val;
    vfloat choices[LVEC];
#pragma unroll
    for (int i = 0; i < LVEC; i++) {
        choices[i] = double_buffer[buffer_id_to_feed_to_sysarr][buffer_row_to_feed_to_sysarr][(buffer_vector_to_feed_to_sysarr / LVEC) * LVEC + i][row];
    }
    val.data = choices[buffer_vector_to_feed_to_sysarr % LVEC];
    val.c = new_row_col_pair & newVal.c;
    return val;
}


// counter is the global counter, which should align with FeederA's counter.
// load_counter is the counter for writing into the feeders,
// which may start at some point in the overall counter range (once FeederA is finished loading)
vfloat
FeederB(n_vfloat newVal,
        vfloat double_buffer[2][X_INTERLEAVED][X_VECS][BANK_X],
        uint load_counter, int col, uint counter) {

    bool write_to_buffer = ((POW2_REM(load_counter, SWAP_RANGE) * LVEC) / (X_INTERLEAVED * X_VECS)) == col;
    // Note: counter is used here because load_counter is not valid if only reading.
    bool buffer_id_to_write_to = (counter / SWAP_RANGE) & 1;
    bool buffer_id_to_feed_to_sysarr = !buffer_id_to_write_to;

    if (write_to_buffer) {
        uchar buffer_vector_to_write_to = POW2_REM(load_counter * LVEC, X_VECS);
        uchar buffer_row_to_write_to = POW2_REM(load_counter * LVEC, X_INTERLEAVED * X_VECS) / X_VECS;
#pragma unroll
        for (int i = 0; i < LVEC; i++) {
            double_buffer[buffer_id_to_write_to][buffer_row_to_write_to][(buffer_vector_to_write_to / LVEC) * LVEC + i][col] = newVal.data[i];
        }
    }

    uchar buffer_row_to_feed_to_sysarr = POW2_REM(counter, X_INTERLEAVED);
    uchar buffer_vector_to_feed_to_sysarr =
        POW2_REM(counter, SWAP_RANGE) / (Y_INTERLEAVED * X_INTERLEAVED);

    vfloat choices[LVEC];
    #pragma unroll
    for (int i = 0; i < LVEC; i++) {
        choices[i] = double_buffer[buffer_id_to_feed_to_sysarr][buffer_row_to_feed_to_sysarr][(buffer_vector_to_feed_to_sysarr / LVEC) * LVEC + i][col];
    }

// Accomodate the floorplanning script
#if LVEC > 1
    return __fpga_reg(choices[buffer_vector_to_feed_to_sysarr % LVEC]);
#else
    return choices[buffer_vector_to_feed_to_sysarr % LVEC];
#endif

}

float
PE(vfloat_bool valA, vfloat valB, float *accum) {
    float oldAcc = __fpga_reg(accum[0]);
    float sum = valA.c ? 0.0f : oldAcc;
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
        sum += valA.data[i] * valB[i];
        // Breaks up dot-8 and larger into dot-4s using fpga_reg.
        // Not needed if VECTOR_SIZE = 4

        // Need dot4 structures to fit the DSP columns on the device
        // (which are 36 DSP long). Dot8 would leave 4 unutilized
        // DSPs.
        #if (FORCE_DOT_4==1) && (VECTOR_SIZE!=4)
            if ((i%4) == 3){
                sum = __fpga_reg(sum);
            }
        #endif
    }

    #pragma unroll
    for (int i = 0; i < SHIFT_REG_SIZE - 1; i++) {
        accum[i] = accum[i + 1];
    }
    accum[SHIFT_REG_SIZE - 1] = sum;
    return oldAcc;
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
void
kernel monolithic() {
    // internal feeder A storage, banked, 1 bank per feeder
    vfloat __attribute__((memory,
                          numbanks(BANK_Y * LVEC),
                          bankwidth(32),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_a[2][Y_INTERLEAVED][X_VECS][BANK_Y];
    // internal feeder B storage, banked, 1 bank per feeder
    vfloat __attribute__((memory,
                          numbanks(BANK_X * LVEC),
                          bankwidth(32),
                          singlepump,
                          max_replicates(1),
                          simple_dual_port)) mem_b[2][X_INTERLEAVED][X_VECS][BANK_X];


#ifdef EMULATE
    for (int i = 0; i < PE_Y; i++) {
        for (int j = 0; j < Y_INTERLEAVED; j++) {
            for (int k = 0; k < X_VECS; k++) {
                mem_a[0][j][k][i] = mem_a[1][j][k][i] = NAN;
            }
        }
    }
    for (int i = 0; i < PE_X; i++) {
        for (int j = 0; j < X_INTERLEAVED; j++) {
            for (int k = 0; k < X_VECS; k++) {
                mem_b[0][j][k][i] = mem_b[1][j][k][i] = -NAN;
            }
        }
    }
#endif

    // internal PE storage for accumulations, ROWS x COLS shift registers
    float accum[PE_Y][PE_X][SHIFT_REG_SIZE];
    // shift register for drain, one per column
    float drain[PE_X][SHIFT_REG_SIZE * (PE_Y - 1) + 1];

    uint counter = 0;
    uint storecount = SHIFT_REG_SIZE * PE_Y;
    uint base = 0;

    const uint n_a_loads = PE_Y * Y_INTERLEAVED * X_VECS / LVEC;
    const uint n_b_loads = PE_X * X_INTERLEAVED * X_VECS / LVEC;
    // Try to load B as late as possible, so that if there is enough time and not enough DDR bandwidth, we
    // can load all of A and then load all of B
    const uint first_b_load = SWAP_RANGE - n_b_loads;

    bool new_row_col_pair = false;
    while (1) {
        vfloat_bool fedA[PE_Y];
        vfloat fedB[PE_X];

        n_vfloat_bool valA;
        n_vfloat valB;
        uint masked_counter = POW2_REM(counter, SWAP_RANGE);
        if (masked_counter < n_a_loads) {
            valA = read_channel_intel(ch_load_a);
            // save latests row_col_pair
            if ((!new_row_col_pair && valA.c) & 0x01)
                base = storecount;
            new_row_col_pair = valA.c;
        }
        // serialize the two reads to reduce burstiness
        if ((masked_counter < first_b_load + n_b_loads) &&
            (masked_counter >= first_b_load))
            valB = read_channel_intel(ch_load_b);

        // private counters for feeder fpga_reg
        uint counterA = counter;

        // recover last known row_col_pair
        valA.c = new_row_col_pair;
        #pragma unroll
        for (int row = 0; row < PE_Y; row++) {
            fedA[row] = FeederA(valA, mem_a, counterA, row);
            #pragma unroll
            for (int i = 0; i < LVEC; i++) {
                valA.data[i] = __fpga_reg(__fpga_reg(valA.data[i]));
            }
            valA.c = __fpga_reg(__fpga_reg(valA.c));
            counterA = __fpga_reg(__fpga_reg(counterA));
        }

        uint counterB = counter;
        #pragma unroll
        for (int x = 0; x < PE_X; x++) {
            // the indexing matches the serialization of the ch_load_b reads
            fedB[x] = FeederB(valB, mem_b, counterB - first_b_load, x, counterB);
            #pragma unroll
            for (int i = 0; i < LVEC; i++) {
                valB.data[i] = __fpga_reg(__fpga_reg(valB.data[i]));
            }
            counterB = __fpga_reg(__fpga_reg(counterB));
        }

        #pragma unroll
        for (int y = 0; y < PE_Y; y++) {
            #pragma unroll
            for (int x = 0; x < PE_X; x++) {
                // compute and store outputs in shift register
                float result = PE(fedA[y], fedB[x], accum[y][x]);
                if (fedA[y].c) {
                    drain[x][y * SHIFT_REG_SIZE] = result;
                }
                fedA[y].data = __fpga_reg(__fpga_reg(fedA[y].data));
                fedA[y].c = __fpga_reg(__fpga_reg(fedA[y].c));
                fedB[x] = __fpga_reg(__fpga_reg(fedB[x]));
            }
        }

        cols_floats results;
        #pragma unroll
        for (int i = 0; i < PE_X; i++) {
            results.data[i] = drain[i][0];

            // Is this code really useful?
            #pragma unroll
            for (int j = 0; j < PE_X; j++) {
                results.data[j] = __fpga_reg(__fpga_reg(results.data[j]));
            }
        }
        if (storecount - base < SHIFT_REG_SIZE * PE_Y)
            write_channel_intel(ch_store_c, results);

        #pragma unroll
        for (int x = 0; x < PE_X; x++) {
            #pragma unroll
            for (int y = 0; y < PE_Y - 1; y++) {
                #pragma unroll
                for (int i = 0; i < SHIFT_REG_SIZE - 1; i++) {
                    drain[x][y * SHIFT_REG_SIZE + i] = drain[x][y * SHIFT_REG_SIZE + i + 1];
                }
                // use fpga_reg at logical PE boundaries - to capture locality
                drain[x][y * SHIFT_REG_SIZE + SHIFT_REG_SIZE - 1] =
                    __fpga_reg(__fpga_reg(drain[x][y * SHIFT_REG_SIZE + SHIFT_REG_SIZE]));
            }
        }
        storecount++;
        counter = POW2_REM(POW2_REM(counter, RANGE) + 1, RANGE);
    }
}


__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
store(global volatile float * restrict C, int c_n_msgs) {
    // We read and discard this many messages
    for (uint i = 0; i < SHIFT_REG_SIZE * PE_Y; i++) {
        read_channel_intel(ch_store_c);
    }


    uint word = 0;
    uchar pos = 0;
    float elems[2 * STORE_WIDTH];
    for (uint i = 0; i < c_n_msgs; i++) {

        uchar crt_pos = POW2_REM(pos, STORE_WIDTH);

        // Align new data
        cols_floats data = read_channel_intel(ch_store_c);
        #pragma unroll
        for (uint j = 0; j < PE_X; j++) {
            elems[j + crt_pos] = data.data[j];
        }

        bool commit = (crt_pos >= STORE_WIDTH - PE_X);
        if (commit) {
#pragma unroll
            for (uint j = 0; j < STORE_WIDTH; j++) {
                C[word * STORE_WIDTH + j] = elems[j];
                elems[j] = elems[STORE_WIDTH + j];
                elems[STORE_WIDTH + j] = 0.0f;
            }
            word++;
        }
        pos += PE_X;
    }
}
