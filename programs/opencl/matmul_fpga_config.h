// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef _PE_SYSTOLIC_ARRAY_H_
#define _PE_SYSTOLIC_ARRAY_H_

// This is important but it is not enforced:
// PE_ROWS + PE_COLS <= ROWS_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define DOT_PROD_VECTOR_SIZE     8

#define FORCE_DOT_4              0

#define PE_Y                     8
#define PE_X                     8

/* #define PE_ROWS                  8 */
/* #define PE_COLS                  8 */

#define ROWS_INTERLEAVED            16
#define COLS_INTERLEAVED            16

#define ACCUM_SHIFT_REG_SIZE        (ROWS_INTERLEAVED * COLS_INTERLEAVED)
#define C_OUT_SHIFT_REG_SIZE        ACCUM_SHIFT_REG_SIZE

#define A_BLOCK_X                   (16 * DOT_PROD_VECTOR_SIZE)
#define A_BLOCK_Y                   (ROWS_INTERLEAVED * PE_Y)

#define B_BLOCK_Y                   A_BLOCK_X
#define B_BLOCK_X                   (COLS_INTERLEAVED * PE_X)

#define C_BLOCK_Y                   A_BLOCK_Y
#define C_BLOCK_X                   B_BLOCK_X

#endif // _PE_SYSTOLIC_ARRAY_H_
