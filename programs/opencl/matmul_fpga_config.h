// Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef PE_SYSTOLIC_ARRAY_H
#define PE_SYSTOLIC_ARRAY_H

// Here are performance figures for the Agilex 7 FPGA I'm working
// with. For N=M=K=8192 matrices:
//
// | VSIZE | PE    | ILEAV | LVEC | SEED | SW  | LAT | FN   | FMAX | TIME  |
// |-------|-------|-------|------|------|-----|-----|------|------|-------|
// | 8     | 8x8   | 16x16 | 1    |      | 16  |     |      |      | 4.93  |
// | 8     | 16x16 | 16x16 | 1    |      | 16  |     |      | 445  | 2.07  |
// | 8     | 16x16 | 16x16 | 1    | 9999 | 16  |     |      | 442  | 2.08  |
// | 8     | 16x16 | 16x16 | 1    | 9998 | 16  |     |      | 460  | 1.32  |
// | 8     | 16x16 | 16x16 | 1    | 9997 | 16  |     |      | 438  | 1.37  |
// | 8     | 16x16 | 16x16 | 1    | 9996 | 16  |     |      | 425  | 1.42  |
// | 8     | 16x16 | 16x16 | 1    | 9995 | 16  |     |      | 431  | 1.39  |
// | 8     | 16x16 | 16x16 | 2    | 9994 | 16  |     |      | 406  | 0.90  |
// | 8     | 16x16 | 16x16 | 4    | 9993 | 16  |     |      | 461  | 32.63 |
// | 8     | 16x16 | 16x16 | 2    | 9992 | 16  |     | (1)  | 456  | 1.24  |
// | 8     | 16x16 | 16x16 | 2    | 9991 | 16  |     | (2)  | 605  | 0.57  |
// | 8     | 16x16 | 16x16 | 2    | 9990 | 128 |     |      | 500  | 0.91  |
// | 8     | 16x16 | 16x16 | 2    | 9989 | 64  |     |      | 500  | 0.89  |
// | 8     | 16x16 | 16x16 | 2    | 9988 | 16  |     |      | 585  | 0.70  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     |      | 485  | 0.82  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     | (3)  | 550  | 0.73  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     | (4)  | 492  | 0.74  |
// | 8     | 16x16 | 16x16 | 2    | 9986 | 16  |     | (4)  | 565  | 0.74  |
// | 8     | 16x16 | 16x16 | 2    | 9985 | 16  |     | (5)  | 565  | 0.58  |
// | 8     | 16x16 | 16x16 | 2    | 9985 | 16  |     | (6)  | -    | -     |
// | 8     | 16x16 | 16x16 | 2    | 9984 | 16  |     |      | 595  | 0.58  |
// | 8     | 16x16 | 16x16 | 2    | 9984 | 16  |     | (7)  | 606  | -     |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  |     | (8)  | 600  | 0.57  |
// | 8     | 16x16 | 16x16 | 4    | 9983 | 16  |     |      | 508  | 28.54 |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  |     | (9)  | 538  | 0.61  |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  |     | (10) | 603  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9982 | 16  |     |      | 606  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9982 | 16  |     |      | 600  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9981 | 16  |     | (11) | 560  | 0.65  |
// | 8     | 16x16 | 16x16 | 2    | 9981 | 16  |     | (12) | 592  | 0.46  |
// | 8     | 32x16 | 32x16 | 2    | 9981 | 16  |     | (13) | 605  | -     |
// | 4     | 16x16 | 16x16 | 2    | 9981 | -   |     | (14) |      |       |
// | 8     | 16x16 | 16x16 | 2    | 9979 | 16  |     |      | 610  | 0.46  |
// | 8     | 16x16 | 16x16 | 2    | 9979 | 16  |     | (15) | 608  | 0.45  |
// | 8     | 16x16 | 16x16 | 4    | 9978 | 16  |     |      | 567  | 0.47  |
// | 8     | 16x16 | 16x16 | 1    | 9978 | 16  |     |      | 600  | 0.47  |
// | 4     | 16x16 | 16x16 | 1    | 9977 | 16  |     |      | 608  | 0.93  |
// | 8     | 16x16 | 16x16 | -    | 9999 | 16  | 89  | (16) | 563  | 0.50  |
// | 8     | 16    | -     | -    | 9970 | 16  | 89  | (17) | 575  | 0.49  |
// | 8     | 16    | -     | -    | 9969 | 16  | 89  | (18) | 588  | 0.48  |
// | 8     | 16    | -     | -    | 9968 | 16  | 89  |      | 585  | 0.48  |
// | 8     | 16    | -     | -    | 9968 | 16  | 89  |      | 585  | 0.48  |
// | 8     | 16    | -     | -    | 9963 | 16  | 88  | (19) | 600  | 0.47  |
//
// LAT = Latency of innermost loop
// SW = I forgot
//
// 1. This refactoring increased the length of the critical chain.
// 2. Reverted last changes.
// 3. No volatile store
// 4. Simpler store kernel
// 5. No volatile
// 6. No FPGA_REGx (breaks Quartus)
// 7. Removed some FPGA_REG2 (causes incorrect results)
// 8. -cl-fast-relaxed-math -cl-mad-enable
// 9. Channel depth 512
// 10. Channel depth 256
// 11. X_SCALE=32
// 12. X_SCALE=8
// 13. Breaks code
// 14. Breaks Quartus
// 15. No interleaving
// 16. No LVEC
// 17. Square SAs
// 18. Simpler counter mgmt
// 19. Simpler handling of the clear signal
//
// This is important but it is not enforced (hmmm):
// PE_X + PE_Y <= Y_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define VECTOR_SIZE             8

// Side length of the square systolic array
#define PE_S                    16

// M must be greater than 128
#define X_SCALE                 8

#define A_BLOCK_X               (X_SCALE * VECTOR_SIZE)
#define A_BLOCK_Y               (PE_S * PE_S)

#define B_BLOCK_Y               A_BLOCK_X
#define B_BLOCK_X               (PE_S * PE_S)

#define C_BLOCK_Y               A_BLOCK_Y
#define C_BLOCK_X               B_BLOCK_X

#endif
