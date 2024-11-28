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

// must be power of 2 smaller than 256
#define WIDTH 16

#define VECTOR_FLOAT4_ZERO          (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO          (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO         (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)

// Number of vectors per block of A
#define A_BLOCK_N_VECTORS           (A_BLOCK_Y * A_BLOCK_X / VECTOR_SIZE)

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

// Vectors in an A block row.
#define X_VECS                    (A_BLOCK_X / VECTOR_SIZE)

#define SWAP_RANGE                  (Y_INTERLEAVED * X_INTERLEAVED * X_VECS)
#define RANGE                       (2 * SWAP_RANGE)

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
loadA(global volatile vfloat* restrict A,
      uint a_n_vectors_in_row_of_blocks,
      uchar a_n_blocks_y,
      uchar b_n_blocks_x) {


    uchar block_col_id = 1;
    // Note: It's important for all counters that wrap around to be
    // zero-based, so that the modN loop recombine algorithm will work
    // If they reset to 1, loop recombine transform needs to
    // conservatively assume that the reset condition is less than the
    // initial value, which would break the transform.
    uint vector_id_in_block = 0;
    uint vector_id_in_row_of_blocks = 0;
    uint vector_id_global = 0;
    uint vector_id_global_row_of_blocks_start = 0;

    uchar row_of_blocks_reuse_counter = 0;

    bool new_row_col_pair = false;
    bool more = true;
    bool feed_zeros_to_flush_last_C_block = false;

    while (more) {
        n_vfloat_bool send_buf;

        #pragma unroll
        for (int i = 0; i < LVEC; i++) {
            if (feed_zeros_to_flush_last_C_block) {
                send_buf.data[i] = VECTOR_ZERO;
                // Under host control, disable loading data from
                // memory to avoid being bandwidth limited. Generate
                // instead numbers in a range above 1.0f, where the
                // actual value is an FP converted uint between
                // 0x3F800000 and 0x3F81000F
            } else {
                send_buf.data[i] = A[vector_id_global * LVEC + i];
            }
        }

        // cast to single-bit bool
        send_buf.c = new_row_col_pair;
        write_channel_intel(ch_load_a, send_buf);

        if (vector_id_in_block == (A_BLOCK_N_VECTORS / LVEC - 1)) {
            vector_id_in_block = 0;
            // we keep new_row_col_pair=true for only the first block
            // in the row of blocks (feeders in the daisy chain expect
            // this functionality)
            new_row_col_pair = false;

        } else {
            vector_id_in_block++;
        }

        vector_id_global++;

        if (vector_id_in_row_of_blocks == A_BLOCK_N_VECTORS / LVEC - 1) {
            // coincides with first block of data being pushed by the feeders
            new_row_col_pair = true;
        }

        if (vector_id_in_row_of_blocks == a_n_vectors_in_row_of_blocks / LVEC - 1) {
            vector_id_in_row_of_blocks = 0;

            if (feed_zeros_to_flush_last_C_block) {
                // we feed only one row of blocks full of zeros to flush the last C block
                more = false;
            }

            // done reusing this row of blocks?
            if (row_of_blocks_reuse_counter == b_n_blocks_x - 1) {

                row_of_blocks_reuse_counter = 0;
                // mark the new start of the row of blocks
                vector_id_global_row_of_blocks_start = vector_id_global;

                // done loading and forwarding the matrix?
                // start feeding zeros to flush last C block
                if (block_col_id == a_n_blocks_y) {
                    feed_zeros_to_flush_last_C_block = true;
                }
                // increment the block_id in the column of blocks
                block_col_id++;

            } else {
                // not done reusing,
                // reset the vector_id_global to the start of row of blocks
                vector_id_global = vector_id_global_row_of_blocks_start;
                row_of_blocks_reuse_counter++;
            }
        } else {
            vector_id_in_row_of_blocks++;
        }
    }
}


__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
loadB(global volatile vfloat* restrict B,
      uint b_n_vectors_in_col_of_blocks,
      uint b_n_vectors_tot,
      uchar a_n_blocks_y) {

    uint vector_id_in_col_of_blocks = 0;
    uint vector_id_global = 0;
    uchar reuse_counter = 0;
    bool feed_zeros = false;

    n_vfloat send_buf;
    while (true) {
        n_vfloat send_buf;
        #pragma unroll
        for (int i = 0; i < LVEC; i++) {
            if (feed_zeros) {
                send_buf.data[i] = VECTOR_ZERO;
            } else {
                send_buf.data[i] = B[vector_id_global * LVEC + i];
            }
        }
        write_channel_intel(ch_load_b, send_buf);

        if (vector_id_in_col_of_blocks == b_n_vectors_in_col_of_blocks / LVEC - 1) {
            vector_id_in_col_of_blocks = 0;
            if (feed_zeros) {
                // we feed only one row of blocks full of zeros to flush the last C block
                break;
            }
        } else {
            vector_id_in_col_of_blocks++;
        }
        if (vector_id_global == b_n_vectors_tot / LVEC - 1) {
            vector_id_global = 0;
            // done reload and forwarding the matrix data?
            if (reuse_counter == a_n_blocks_y-1) {
                feed_zeros = true;
            }
            reuse_counter++;
        } else {
            vector_id_global++;
        }
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
        for (int col = 0; col < PE_X; col++) {
            // the indexing matches the serialization of the ch_load_b reads
            fedB[col] = FeederB(valB, mem_b, counterB - first_b_load, col, counterB);
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
store(global volatile float * restrict C, int c_n_coalesced_words) {
    bool more = true;

    int word = 0;
    uchar pos = 0;
    int i = 0;

    float elems[2 * WIDTH] = {0.0f};
    while (more) {
        float tmpelems[2 * WIDTH] __attribute__((register));
        // Clear space where we build the aligned word
        #pragma unroll
        for (int j = 0; j < 2 * WIDTH; j++) {
            tmpelems[j] = 0.0f;
        }

        cols_floats root_data = read_channel_intel(ch_store_c);
        more = i < c_n_coalesced_words + SHIFT_REG_SIZE * PE_Y - 1;

        uchar crt_pos = POW2_REM(pos, WIDTH);
        bool commit = (crt_pos >= WIDTH - PE_X) || !more;

        // Align new data
        #pragma unroll
        for (int j = 0; j < PE_X; j++) {
            tmpelems[j + crt_pos] = i >= SHIFT_REG_SIZE * PE_Y ? root_data.data[j] : 0.0f;
        }
        // Merge with old data
        #pragma unroll
        for (int j = 0; j < 2 * WIDTH; j++) {
            elems[j] = as_float(as_uint(elems[j]) | as_uint(tmpelems[j]));
        }

        if (commit && i >= SHIFT_REG_SIZE * PE_Y) {
#pragma unroll
            for (int j = 0; j < WIDTH; j++) {
                C[word * WIDTH + j] = elems[j];
                elems[j] = elems[WIDTH + j];
                elems[WIDTH + j] = 0.0f;
            }
            word++;
        }
        pos += i >= SHIFT_REG_SIZE * PE_Y ? PE_X : 0;
        i++;
    }
}
