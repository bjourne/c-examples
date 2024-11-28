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

#define VEC                         DOT_PROD_VECTOR_SIZE
#define ROWS                        PE_ROWS
#define COLS                        PE_COLS
#define COLS_INTERLEAVED            COLUMNS_INTERLEAVED

// The number of rows rounded up to the next power of 2
#if ROWS <= 1
#define BANK_Y 1
#elif ROWS <= 2
#define BANK_Y 2
#elif ROWS <= 4
#define BANK_Y 4
#elif ROWS <= 8
#define BANK_Y 8
#elif ROWS <= 16
#define BANK_Y 16
#elif ROWS <= 32
#define BANK_Y 32
#elif ROWS <= 64
#define BANK_Y 64
#elif ROWS <= 128
#define BANK_Y 128
#elif ROWS <= 256
#define BANK_Y 256
#else
#error "ROWS too large, BANK_Y cannot be defined"
#endif

// The number of columns rounded up to the next power of 2
#if COLS <= 1
#define BANK_X 1
#elif COLS <= 2
#define BANK_X 2
#elif COLS <= 4
#define BANK_X 4
#elif COLS <= 8
#define BANK_X 8
#elif COLS <= 16
#define BANK_X 16
#elif COLS <= 32
#define BANK_X 32
#elif COLS <= 64
#define BANK_X 64
#elif COLS <= 128
#define BANK_X 128
#elif COLS <= 256
#define BANK_X 256
#else
#error "COLS too large, BANK_X cannot be defined"
#endif


// Defines the width of channels in number of vectors.
#ifndef LVEC
#define LVEC 1
#endif

#define ROW_VECS                    (A_BLOCK_X / VEC)
#define ROW_VECS_MASK               (ROW_VECS - 1)

#define SWAP_RANGE                  (ROWS_INTERLEAVED * COLS_INTERLEAVED * ROW_VECS)
#define SWAP_RANGE_MASK             (SWAP_RANGE - 1)

#define RANGE                       (2 * SWAP_RANGE)
#define RANGE_MASK                  (RANGE - 1)

////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////
#if DOT_PROD_VECTOR_SIZE==4
typedef float4 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT4_ZERO
#elif DOT_PROD_VECTOR_SIZE==8
typedef float8 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT8_ZERO
#elif DOT_PROD_VECTOR_SIZE==16
typedef float16 vfloat;
#define VECTOR_ZERO         VECTOR_FLOAT16_ZERO
#else
#error Unsupported DOT_PROD_VECTOR_SIZE
#endif

////////////////////////////////////////////////////////////////////////
// Structs
////////////////////////////////////////////////////////////////////////
typedef struct {
    float drain_data[PE_COLS];
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
channel n_vfloat_bool loadAChannel __attribute__((depth(64)));
channel n_vfloat loadBChannel __attribute__((depth(64)));
channel cols_floats storeCChannel __attribute__((depth(64)));


__attribute__((max_global_work_dim(0)))
kernel void
loadA(global volatile vfloat* restrict A,
      uint mat_a_num_vectors_in_row_of_blocks,
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
    bool more_vectors_to_load_and_forward = true;
    bool feed_zeros_to_flush_last_C_block = false;

    while(more_vectors_to_load_and_forward) {

        n_vfloat_bool A_local;

        #pragma unroll
        for (int i = 0; i < LVEC; i++) {
            if (feed_zeros_to_flush_last_C_block) {
                A_local.data[i] = VECTOR_ZERO;
                // Under host control, disable loading data from
                // memory to avoid being bandwidth limited. Generate
                // instead numbers in a range above 1.0f, where the
                // actual value is an FP converted uint between
                // 0x3F800000 and 0x3F81000F
            } else {
                A_local.data[i] = A[vector_id_global * LVEC + i];
            }
        }

        A_local.c = new_row_col_pair; // cast to single-bit bool
        write_channel_intel(loadAChannel, A_local);

        if (vector_id_in_block == (MAT_A_BLOCK_NUM_VECTORS / LVEC - 1)) {
            vector_id_in_block = 0;
            // we keep new_row_col_pair=true for only the first block
            // in the row of blocks (feeders in the daisy chain expect
            // this functionality)
            new_row_col_pair = false;

        } else {
            vector_id_in_block++;
        }

        vector_id_global++;

        if (vector_id_in_row_of_blocks == MAT_A_BLOCK_NUM_VECTORS / LVEC - 1) {
            // coincides with first block of data being pushed by the feeders
            new_row_col_pair = true;
        }

        if (vector_id_in_row_of_blocks == mat_a_num_vectors_in_row_of_blocks / LVEC - 1) {
            vector_id_in_row_of_blocks = 0;

            if (feed_zeros_to_flush_last_C_block) {
                // we feed only one row of blocks full of zeros to flush the last C block
                more_vectors_to_load_and_forward = false;
            }

            // done reusing this row of blocks?
            if (row_of_blocks_reuse_counter == b_n_blocks_x-1) {

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
kernel void
loadB(global volatile vfloat* restrict B,
      uint mat_b_num_vectors_in_col_of_blocks,
      uint mat_b_num_vectors_in_matrix,
      uchar a_n_blocks_y) {

    uint vector_id_in_col_of_blocks = 0;
    uint vector_id_global = 0;
    uchar matrix_B_reuse_counter = 0;

    bool more_vectors_to_load_and_forward = true;
    bool feed_zeros_to_flush_last_C_block = false;

    while(more_vectors_to_load_and_forward) {

        n_vfloat B_local;

        #pragma unroll
        for (int i = 0; i < LVEC; i++) {
            if (feed_zeros_to_flush_last_C_block) {
                B_local.data[i] = VECTOR_ZERO;
            } else {
                B_local.data[i] = B[vector_id_global * LVEC + i];
            }
        }

        write_channel_intel(loadBChannel, B_local);

        if (vector_id_in_col_of_blocks == mat_b_num_vectors_in_col_of_blocks / LVEC - 1) {
            vector_id_in_col_of_blocks = 0;
            if (feed_zeros_to_flush_last_C_block) {
                // we feed only one row of blocks full of zeros to flush the last C block
                more_vectors_to_load_and_forward = false;
            }
        } else {
            vector_id_in_col_of_blocks++;
        }

        if (vector_id_global == mat_b_num_vectors_in_matrix / LVEC - 1) {
            vector_id_global = 0;
            // done reload and forwarding the matrix data?
            if (matrix_B_reuse_counter == a_n_blocks_y-1) {
                feed_zeros_to_flush_last_C_block = true;
            }
            matrix_B_reuse_counter++;
        } else {
            vector_id_global++;
        }

    }
}



vfloat_bool
FeederA(n_vfloat_bool newVal,
        vfloat matrix_block_double_buffer[2][ROWS_INTERLEAVED][ROW_VECS][BANK_Y],
        uint counter, int row) {

    bool write_to_buffer =
        ((counter & SWAP_RANGE_MASK) * LVEC / (ROWS_INTERLEAVED * ROW_VECS)) == row;
    bool new_row_col_pair =
        (counter & SWAP_RANGE_MASK) < (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED);
    bool buffer_id_to_write_to = (counter / SWAP_RANGE) & 1;
    bool buffer_id_to_feed_to_sysarr = !buffer_id_to_write_to;

    if (write_to_buffer) {
        uchar buffer_vector_to_write_to = (counter * LVEC) & ROW_VECS_MASK;
        uchar buffer_row_to_write_to = ((counter * LVEC) & ((ROWS_INTERLEAVED * ROW_VECS)-1)) / ROW_VECS;
        #pragma unroll
        for (int i = 0; i < LVEC; i++) {
            matrix_block_double_buffer[buffer_id_to_write_to][buffer_row_to_write_to][(buffer_vector_to_write_to / LVEC) * LVEC + i][row] = newVal.data[i];
        }
    }

    uchar buffer_row_to_feed_to_sysarr =
        (counter & ((ROWS_INTERLEAVED * COLUMNS_INTERLEAVED)-1)) / COLUMNS_INTERLEAVED;
    uchar buffer_vector_to_feed_to_sysarr =
        (counter & SWAP_RANGE_MASK) / (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED);

    vfloat_bool val;
    vfloat choices[LVEC];
    #pragma unroll
    for (int i = 0; i < LVEC; i++) {
        choices[i] = matrix_block_double_buffer[buffer_id_to_feed_to_sysarr][buffer_row_to_feed_to_sysarr][(buffer_vector_to_feed_to_sysarr / LVEC) * LVEC + i][row];
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
        vfloat matrix_block_double_buffer[2][COLUMNS_INTERLEAVED][ROW_VECS][BANK_X],
        uint load_counter, int col, uint counter) {

    bool write_to_buffer =
        (((load_counter & SWAP_RANGE_MASK) * LVEC) / (COLUMNS_INTERLEAVED * ROW_VECS)) == col;
    // Note: counter is used here because load_counter is not valid if only reading.
    bool buffer_id_to_write_to = (counter / SWAP_RANGE) & 1;
    bool buffer_id_to_feed_to_sysarr = !buffer_id_to_write_to;

    if (write_to_buffer) {
        uchar buffer_vector_to_write_to = (load_counter * LVEC) & ROW_VECS_MASK;
        uchar buffer_row_to_write_to = ((load_counter * LVEC) & ((COLUMNS_INTERLEAVED * ROW_VECS)-1)) / ROW_VECS;
#pragma unroll
        for (int i = 0; i < LVEC; i++) {
            matrix_block_double_buffer[buffer_id_to_write_to][buffer_row_to_write_to][(buffer_vector_to_write_to / LVEC) * LVEC + i][col] = newVal.data[i];
        }
    }

    uchar buffer_row_to_feed_to_sysarr =
        counter & (COLUMNS_INTERLEAVED - 1);
    uchar buffer_vector_to_feed_to_sysarr =
        (counter & SWAP_RANGE_MASK) / (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED);

    vfloat choices[LVEC];
    #pragma unroll
    for (int i = 0; i < LVEC; i++) {
        choices[i] = matrix_block_double_buffer[buffer_id_to_feed_to_sysarr][buffer_row_to_feed_to_sysarr][(buffer_vector_to_feed_to_sysarr / LVEC) * LVEC + i][col];
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
    for (int i = 0; i < DOT_PROD_VECTOR_SIZE; i++) {
        sum += valA.data[i] * valB[i];
        // Breaks up dot-8 and larger into dot-4s using fpga_reg.
        // Not needed if DOT_PROD_VECTOR_SIZE = 4

        // Need dot4 structures to fit the DSP columns on the device
        // (which are 36 DSP long). Dot8 would leave 4 unutilized
        // DSPs.
        #if (FORCE_DOT_4==1) && (DOT_PROD_VECTOR_SIZE!=4)
            if ((i%4) == 3){
                sum = __fpga_reg(sum);
            }
        #endif
    }

    #pragma unroll
    for (int i = 0; i < ACCUM_SHIFT_REG_SIZE - 1; i++) {
        accum[i] = accum[i + 1];
    }
    accum[ACCUM_SHIFT_REG_SIZE - 1] = sum;
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
                               simple_dual_port)) memA[2][ROWS_INTERLEAVED][ROW_VECS][BANK_Y];
    // internal feeder B storage, banked, 1 bank per feeder
    vfloat __attribute__((memory,
                               numbanks(BANK_X * LVEC),
                               bankwidth(32),
                               singlepump,
                               max_replicates(1),
                               simple_dual_port)) memB[2][COLUMNS_INTERLEAVED][ROW_VECS][BANK_X];


#ifdef EMULATE
    for (int i = 0; i < PE_ROWS; i++) {
        for (int j = 0; j < ROWS_INTERLEAVED; j++) {
            for (int k = 0; k < ROW_VECS; k++) {
                memA[0][j][k][i] = memA[1][j][k][i] = NAN;
            }
        }
    }
    for (int i = 0; i < PE_COLS; i++) {
        for (int j = 0; j < COLUMNS_INTERLEAVED; j++) {
            for (int k = 0; k < ROW_VECS; k++) {
                memB[0][j][k][i] = memB[1][j][k][i] = -NAN;
            }
        }
    }
#endif

    // internal PE storage for accumulations, ROWS x COLS shift registers
    float accum[PE_ROWS][PE_COLS][ACCUM_SHIFT_REG_SIZE];
    // shift register for drain, one per column
    float drain[PE_COLS][ACCUM_SHIFT_REG_SIZE * (PE_ROWS - 1) + 1];

    uint counter = 0;
    uint storecount = ACCUM_SHIFT_REG_SIZE * PE_ROWS;
    uint base = 0;

    const uint num_a_loads = PE_ROWS * ROWS_INTERLEAVED * ROW_VECS / LVEC;
    const uint num_b_loads = PE_COLS * COLUMNS_INTERLEAVED * ROW_VECS / LVEC;
    // Try to load B as late as possible, so that if there is enough time and not enough DDR bandwidth, we
    // can load all of A and then load all of B
    const uint first_b_load = SWAP_RANGE - num_b_loads;

    bool new_row_col_pair = false;
    while (1) {
        vfloat_bool fedA[PE_ROWS];
        vfloat fedB[PE_COLS];

        n_vfloat_bool valA;
        n_vfloat valB;
        if ((counter & SWAP_RANGE_MASK) < num_a_loads) {
            valA = read_channel_intel(loadAChannel);
            // save latests row_col_pair
            if ((!new_row_col_pair && valA.c) & 0x01)
                base = storecount;
            new_row_col_pair = valA.c;
        }
        // serialize the two reads to reduce burstiness
        if (((counter & SWAP_RANGE_MASK) < first_b_load + num_b_loads) &&
            ((counter & SWAP_RANGE_MASK) >= first_b_load))
            valB = read_channel_intel(loadBChannel);

        // private counters for feeder fpga_reg
        uint counterA = counter;

        // recover last known row_col_pair
        valA.c = new_row_col_pair;
        #pragma unroll
        for (int row = 0; row < PE_ROWS; row++) {
            fedA[row] = FeederA(valA, memA, counterA, row);
            #pragma unroll
            for (int i = 0; i < LVEC; i++) {
                valA.data[i] = __fpga_reg(__fpga_reg(valA.data[i]));
            }
            valA.c = __fpga_reg(__fpga_reg(valA.c));
            counterA = __fpga_reg(__fpga_reg(counterA));
        }

        uint counterB = counter;
        #pragma unroll
        for (int col = 0; col < PE_COLS; col++) {
            // the indexing matches the serialization of the loadBChannel reads
            fedB[col] = FeederB(valB, memB, counterB - first_b_load, col, counterB);
            #pragma unroll
            for (int i = 0; i < LVEC; i++) {
                valB.data[i] = __fpga_reg(__fpga_reg(valB.data[i]));
            }
            counterB = __fpga_reg(__fpga_reg(counterB));
        }

        #pragma unroll
        for (int row = 0; row < PE_ROWS; row++) {
            #pragma unroll
            for (int col = 0; col < PE_COLS; col++) {
                // compute and store outputs in shift register
                float result =  PE(fedA[row], fedB[col], accum[row][col]);
                if (fedA[row].c) {
                    drain[col][row * ACCUM_SHIFT_REG_SIZE] = result;
                }
                fedA[row].data = __fpga_reg(__fpga_reg(fedA[row].data));
                fedA[row].c = __fpga_reg(__fpga_reg(fedA[row].c));
                fedB[col] = __fpga_reg(__fpga_reg(fedB[col]));
            }
        }

        cols_floats results;
        #pragma unroll
        for (int col = 0; col < PE_COLS; col++) {
            results.drain_data[col] = drain[col][0];
            #pragma unroll
            for (int i = 0; i < PE_COLS; i++) {
                results.drain_data[i] = __fpga_reg(__fpga_reg(results.drain_data[i]));
            }
        }
        if (storecount - base < ACCUM_SHIFT_REG_SIZE * PE_ROWS)
            write_channel_intel(storeCChannel, results);

        #pragma unroll
        for (int col = 0; col < PE_COLS; col++) {
            #pragma unroll
            for (int row = 0; row < PE_ROWS - 1; row++) {
                #pragma unroll
                for (int i = 0; i < ACCUM_SHIFT_REG_SIZE - 1; i++) {
                    drain[col][row * ACCUM_SHIFT_REG_SIZE + i] = drain[col][row * ACCUM_SHIFT_REG_SIZE + i + 1];
                }
                // use fpga_reg at logical PE boundaries - to capture locality
                drain[col][row * ACCUM_SHIFT_REG_SIZE + ACCUM_SHIFT_REG_SIZE - 1] =
                    __fpga_reg(__fpga_reg(drain[col][row * ACCUM_SHIFT_REG_SIZE + ACCUM_SHIFT_REG_SIZE]));
            }
        }
        storecount++;
        counter = ((counter & RANGE_MASK) + 1) & RANGE_MASK;
    }
}


__attribute__((max_global_work_dim(0)))
kernel void
store(
    global volatile float * restrict C,
    int mat_c_num_coalesced_words
) {
    bool more_words_to_write = true;

    int word = 0;
    uchar pos = 0;
    int i = 0;

    float elems[2 * WIDTH] = {0.0f};

    while (more_words_to_write) {
        float tmpelems[2 * WIDTH] __attribute__((register));
        // Clear space where we build the aligned word
        #pragma unroll
        for (int j = 0; j < 2 * WIDTH; j++) {
            tmpelems[j] = 0.0f;
        }

        cols_floats root_data = read_channel_intel(storeCChannel);
        more_words_to_write = i < mat_c_num_coalesced_words + ACCUM_SHIFT_REG_SIZE * PE_ROWS - 1;

        uchar crt_pos = pos & (WIDTH - 1);
        bool commit = (crt_pos >= WIDTH - PE_COLS) || !more_words_to_write;
        pos += i < ACCUM_SHIFT_REG_SIZE * PE_ROWS ? 0 : PE_COLS;

        // Align new data
        #pragma unroll
        for (int j = 0; j < PE_COLS; j++) {
            tmpelems[j + crt_pos] = i >= ACCUM_SHIFT_REG_SIZE * PE_ROWS ? root_data.drain_data[j] : 0.0f;
        }
        // Merge with old data
        #pragma unroll
        for (int j = 0; j < 2 * WIDTH; j++) {
            elems[j] = as_float(as_uint(elems[j]) | as_uint(tmpelems[j]));
        }

        if (commit && i >= ACCUM_SHIFT_REG_SIZE * PE_ROWS) {
        #pragma unroll
        for (int j = 0; j < WIDTH; j++) {
            C[word * WIDTH + j] = elems[j];
            elems[j] = elems[WIDTH + j];
            elems[WIDTH + j] = 0.0f;
            }
            word++;
        }
        i++;
    }
}
