// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef _PE_SYSTOLIC_ARRAY_H_
#define _PE_SYSTOLIC_ARRAY_H_

// This is important but it is not enforced:
// PE_ROWS + PE_COLS <= ROWS_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define VECTOR_SIZE     16

#define FORCE_DOT_4              0

// Must be powers of two
#define PE_Y                     16
#define PE_X                     16

// Must be powers of two
#define Y_INTERLEAVED            16
#define X_INTERLEAVED            16

#define A_BLOCK_X                   (16 * VECTOR_SIZE)
#define A_BLOCK_Y                   (Y_INTERLEAVED * PE_Y)

#define B_BLOCK_Y                   A_BLOCK_X
#define B_BLOCK_X                   (X_INTERLEAVED * PE_X)

#define C_BLOCK_Y                   A_BLOCK_Y
#define C_BLOCK_X                   B_BLOCK_X

#endif // _PE_SYSTOLIC_ARRAY_H_
