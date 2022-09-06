// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_DCT_H
#define TENSORS_DCT_H

#include "tensors/tensors.h"

// Constants for 8 point dct. Are these long names really necessary?
// Yes they are.
#define TENSOR_DCT8_LOEFFLER_C1_A \
    -0.78569495838710218127789736765721679604624211314138128724
#define TENSOR_DCT8_LOEFFLER_C1_B \
    -1.17587560241935871697446710461126127790162534864529090275
#define TENSOR_DCT8_LOEFFLER_C1_C \
    0.980785280403230449126182236134239036973933730893336095002
#define TENSOR_DCT8_LOEFFLER_C3_A \
    -0.27589937928294301233595756366937288236362362123244591752
#define TENSOR_DCT8_LOEFFLER_C3_B \
    -1.38703984532214746182161919156643863111349800274205400937
#define TENSOR_DCT8_LOEFFLER_C3_C \
    0.831469612302545237078788377617905756738560811987249963446
#define TENSOR_DCT8_LOEFFLER_C6_A \
    0.765366864730179543456919968060797733522689124971254082867
#define TENSOR_DCT8_LOEFFLER_C6_B \
    -1.84775906502257351225636637879357657364483325172728497223
#define TENSOR_DCT8_LOEFFLER_C6_C \
    0.541196100146196984399723205366389420061072063378015444681

#define TENSOR_DCT8_LOEFFLER_NORM1      0.35355339059327373f
#define TENSOR_DCT8_LOEFFLER_NORM2      0.5

inline void
tensor_dct8_loeffler(float x[8], float y[8]) {

    // Stage 1 reflectors
    float s10 = x[0] + x[7];
    float s11 = x[1] + x[6];
    float s12 = x[2] + x[5];
    float s13 = x[3] + x[4];

    float s14 = x[3] - x[4];
    float s15 = x[2] - x[5];
    float s16 = x[1] - x[6];
    float s17 = x[0] - x[7];

    // Stage 2 stuff
    float s20 = s10 + s13;
    float s21 = s11 + s12;
    float s22 = s11 - s12;
    float s23 = s10 - s13;

    // Rotations?
    float c3_rot_tmp = TENSOR_DCT8_LOEFFLER_C3_C * (s14 + s17);
    float c1_rot_tmp = TENSOR_DCT8_LOEFFLER_C1_C * (s15 + s16);

    float s24 = TENSOR_DCT8_LOEFFLER_C3_A * s17 + c3_rot_tmp;
    float s25 = TENSOR_DCT8_LOEFFLER_C1_A * s16 + c1_rot_tmp;
    float s26 = TENSOR_DCT8_LOEFFLER_C1_B * s15 + c1_rot_tmp;
    float s27 = TENSOR_DCT8_LOEFFLER_C3_B * s14 + c3_rot_tmp;

    // Stage 3
    float c6_rot_tmp = TENSOR_DCT8_LOEFFLER_C6_C * (s22 + s23);

    float s30 = s20 + s21;
    float s31 = s20 - s21;
    float s34 = s24 + s26;
    float s35 = s27 - s25;
    float s36 = s24 - s26;
    float s37 = s27 + s25;

    y[0] = TENSOR_DCT8_LOEFFLER_NORM1 * s30;
    y[1] = TENSOR_DCT8_LOEFFLER_NORM1 * (s37 + s34);
    y[2] = TENSOR_DCT8_LOEFFLER_NORM1 * (TENSOR_DCT8_LOEFFLER_C6_A * s23 + c6_rot_tmp);
    y[3] = TENSOR_DCT8_LOEFFLER_NORM2 * s35;
    y[4] = TENSOR_DCT8_LOEFFLER_NORM1 * s31;
    y[5] = TENSOR_DCT8_LOEFFLER_NORM2 * s36;
    y[6] = TENSOR_DCT8_LOEFFLER_NORM1 * (TENSOR_DCT8_LOEFFLER_C6_B * s22 + c6_rot_tmp);
    y[7] = TENSOR_DCT8_LOEFFLER_NORM1 * (s37 - s34);
}

#define TENSOR_DCT8_NVIDIA_CA 1.3870398453221474618216191915664f
#define TENSOR_DCT8_NVIDIA_CB 1.3065629648763765278566431734272f
#define TENSOR_DCT8_NVIDIA_CC 1.1758756024193587169744671046113f
#define TENSOR_DCT8_NVIDIA_CD 0.78569495838710218127789736765722f
#define TENSOR_DCT8_NVIDIA_CE 0.54119610014619698439972320536639f
#define TENSOR_DCT8_NVIDIA_CF 0.27589937928294301233595756366937f

#define TENSOR_DCT8_NVIDIA_NORM 0.35355339059327376220042218105242f

// Nvidia's variant of the algorithm.
inline void
tensor_dct8_nvidia(float x[8], float y[8]) {

    // Stage 1 reflectors
    float s10 = x[0] + x[7];
    float s11 = x[1] + x[6];
    float s12 = x[2] + x[5];
    float s13 = x[3] + x[4];
    float s14 = x[0] - x[7];
    float s15 = x[2] - x[5];
    float s16 = x[4] - x[3];
    float s17 = x[6] - x[1];

    // Stage 2 things
    float s20 = s10 + s13;
    float s21 = s10 - s13;
    float s22 = s11 + s12;
    float s23 = s11 - s12;

    float norm = TENSOR_DCT8_NVIDIA_NORM;

    y[0] = norm * (s20 + s22);
    y[2] = norm * (TENSOR_DCT8_NVIDIA_CB * s21 + TENSOR_DCT8_NVIDIA_CE * s23);
    y[4] = norm * (s20 - s22);
    y[6] = norm * (TENSOR_DCT8_NVIDIA_CE * s21 - TENSOR_DCT8_NVIDIA_CB * s23);
    y[1] = norm * (
        TENSOR_DCT8_NVIDIA_CA * s14 -
        TENSOR_DCT8_NVIDIA_CC * s17 +
        TENSOR_DCT8_NVIDIA_CD * s15 -
        TENSOR_DCT8_NVIDIA_CF * s16
    );
    y[3] = norm * (
        TENSOR_DCT8_NVIDIA_CC * s14 +
        TENSOR_DCT8_NVIDIA_CF * s17 -
        TENSOR_DCT8_NVIDIA_CA * s15 +
        TENSOR_DCT8_NVIDIA_CD * s16
    );
    y[5] = norm * (
        TENSOR_DCT8_NVIDIA_CD * s14 +
        TENSOR_DCT8_NVIDIA_CA * s17 +
        TENSOR_DCT8_NVIDIA_CF * s15 -
        TENSOR_DCT8_NVIDIA_CC * s16
    );
    y[7] = norm * (
        TENSOR_DCT8_NVIDIA_CF * s14 +
        TENSOR_DCT8_NVIDIA_CD * s17 +
        TENSOR_DCT8_NVIDIA_CC * s15 +
        TENSOR_DCT8_NVIDIA_CA * s16
    );
}


void tensor_dct2d_rect(tensor *src, tensor *dst,
                       int sy, int sx, int height, int width);

void tensor_idct2d(tensor *src, tensor *dst);

// DCT in blocks
void tensor_dct2d_blocks(tensor *src, tensor *dst,
                         int block_height, int block_width);
void tensor_dct2d_8x8_blocks_loeffler(tensor *src, tensor *dst);



#endif
