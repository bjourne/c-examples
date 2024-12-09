// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef PE_SYSTOLIC_ARRAY_H
#define PE_SYSTOLIC_ARRAY_H

// Here are performance figures for the Agilex 7 FPGA I'm working
// with. For N=M=K=8192 matrices:
//
// | VSIZE | PE    | ILEAV | LVEC | SEED | SW  | FN  | FMAX | TIME  |
// |-------|-------|-------|------|------|-----|-----|------|-------|
// | 8     | 8x8   | 16x16 | 1    |      | 16  |     |      | 4.93  |
// | 8     | 16x16 | 16x16 | 1    |      | 16  |     | 445  | 2.07  |
// | 8     | 16x16 | 16x16 | 1    | 9999 | 16  |     | 442  | 2.08  |
// | 8     | 16x16 | 16x16 | 1    | 9998 | 16  |     | 460  | 1.32  |
// | 8     | 16x16 | 16x16 | 1    | 9997 | 16  |     | 438  | 1.37  |
// | 8     | 16x16 | 16x16 | 1    | 9996 | 16  |     | 425  | 1.42  |
// | 8     | 16x16 | 16x16 | 1    | 9995 | 16  |     | 431  | 1.39  |
// | 8     | 16x16 | 16x16 | 2    | 9994 | 16  |     | 406  | 0.90  |
// | 8     | 16x16 | 16x16 | 4    | 9993 | 16  |     | 461  | 32.63 |
// | 8     | 16x16 | 16x16 | 2    | 9992 | 16  | (1) | 456  | 1.24  |
// | 8     | 16x16 | 16x16 | 2    | 9991 | 16  | (2) | 605  | 0.57  |
// | 8     | 16x16 | 16x16 | 2    | 9990 | 128 |     | 500  | 0.91  |
// | 8     | 16x16 | 16x16 | 2    | 9989 | 64  |     | 500  | 0.89  |
// | 8     | 16x16 | 16x16 | 2    | 9988 | 16  |     | 585  | 0.70  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     | 485  | 0.82  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  | (3) | 550  | 0.73  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  | (4) | 492  | 0.74  |
// | 8     | 16x16 | 16x16 | 2    | 9986 | 16  | (4) | 565  | 0.74  |
// | 8     | 16x16 | 16x16 | 2    | 9985 | 16  | (5) | 565  | 0.58  |
// | 8     | 16x16 | 16x16 | 2    | 9985 | 16  | (6) | -    | -     |
// | 8     | 16x16 | 16x16 | 2    | 9984 | 16  |     | 595  | 0.58  |
// | 8     | 16x16 | 16x16 | 2    | 9984 | 16  | (7) | 606  | -     |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  | (8) | 600  | 0.57  |
// | 8     | 16x16 | 16x16 | 4    | 9983 | 16  |     | 508  | 28.54 |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  | (9) | 538  | 0.61  |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  | (10)| 603  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9982 | 16  |     | 606  | 0.55  |
//
// 1. This refactoring increased the length of the critical chain.
// 2. Reverted last changes.
// 3. No volatile store
// 4. Simpler store kernel
// 5. No volatile
// 6. No FPGA_REGx (it broke Quartus)
// 7. Removed some FPGA_REG2 (causes incorrect results)
// 8. -cl-fast-relaxed-math -cl-mad-enable
// 9. Channel depth 512
// 10. Channel depth 256
//
// This is important but it is not enforced:
// PE_X + PE_Y <= Y_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define VECTOR_SIZE             8

#define FORCE_DOT_4             0

#define PE_Y                    16
#define PE_X                    16

// Interleaving. Must be powers of two
#define Y_ILEAVE                16
#define X_ILEAVE                16

#define X_SCALE                 16

#define A_BLOCK_X               (X_SCALE * VECTOR_SIZE)
#define A_BLOCK_Y               (Y_ILEAVE * PE_Y)

#define B_BLOCK_Y               A_BLOCK_X
#define B_BLOCK_X               (X_ILEAVE * PE_X)

#define C_BLOCK_Y               A_BLOCK_Y
#define C_BLOCK_X               B_BLOCK_X

#endif
