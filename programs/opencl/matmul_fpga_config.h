// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef _PE_SYSTOLIC_ARRAY_H_
#define _PE_SYSTOLIC_ARRAY_H_

// This is important but it is not enforced:
// PE_ROWS + PE_COLS <= ROWS_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define DOT_PROD_VECTOR_SIZE     8

#define FORCE_DOT_4              0

#define PE_ROWS                  8
#define PE_COLS                  8

#define ROWS_INTERLEAVED         16

#define COLUMNS_INTERLEAVED      16

#define ACCUM_SHIFT_REG_SIZE        (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED)
#define C_OUT_SHIFT_REG_SIZE        ACCUM_SHIFT_REG_SIZE

#define A_BLOCK_X                   (16 * DOT_PROD_VECTOR_SIZE)
#define A_BLOCK_Y                   (ROWS_INTERLEAVED * PE_ROWS)
#define A_BLOCK_SIZE                (A_BLOCK_Y * A_BLOCK_X)

// Number of vectors per block of A
#define A_BLOCK_N_VECTORS           (A_BLOCK_SIZE / DOT_PROD_VECTOR_SIZE)

#define B_BLOCK_Y                   A_BLOCK_X
#define B_BLOCK_X                   (COLUMNS_INTERLEAVED * PE_COLS)
#define B_BLOCK_SIZE                (B_BLOCK_Y * B_BLOCK_X)

#define C_BLOCK_Y                   A_BLOCK_Y
#define C_BLOCK_X                   B_BLOCK_X

#endif // _PE_SYSTOLIC_ARRAY_H_
