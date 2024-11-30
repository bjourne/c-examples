// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef _PE_SYSTOLIC_ARRAY_H_
#define _PE_SYSTOLIC_ARRAY_H_

// Here are performance figures for the FPGA I'm working with. For
// square 8192x8192 matrices:
//
// | FMAX | VSIZE | PE    | INTER | LVEC | SEED | TIME |
// |------|-------|-------|-------|------|------|------|
// |      | 8     | 8x8   | 16x16 | 1    |      | 4.93 |
// | 445  | 8     | 16x16 | 16x16 | 1    |      | 2.07 |
// | 442  | 8     | 16x16 | 16x16 | 1    | 9999 | 2.08 |
// | 460  | 8     | 16x16 | 16x16 | 1    | 9998 | 1.32 |
// | 438  | 8     | 16x16 | 16x16 | 1    | 9997 | 1.37 |
// | 425  | 8     | 16x16 | 16x16 | 1    | 9996 | 1.42 |
// |      | 8     | 16x16 | 16x16 | 1    | 9995 |      |

// This is important but it is not enforced:
// PE_X + PE_Y <= Y_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define VECTOR_SIZE             8

#define FORCE_DOT_4              0

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
