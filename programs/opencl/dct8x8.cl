// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#define BLOCK_SIZE      8

// a = sqrt(2) * cos(1 * pi / 16)
#define    C_a 1.38703984532214746182161919156640f
// b = sqrt(2) * cos(2 * pi / 16)
#define    C_b 1.30656296487637652785664317342720f
// c = sqrt(2) * cos(3 * pi / 16)
#define    C_c 1.17587560241935871697446710461130f
// d = sqrt(2) * cos(5 * pi / 16)
#define    C_d 0.78569495838710218127789736765722f
// e = sqrt(2) * cos(6 * pi / 16)
#define    C_e 0.54119610014619698439972320536639f
// f = sqrt(2) * cos(7 * pi / 16)
#define    C_f 0.27589937928294301233595756366937f
// norm = 1 / sqrt(8)
#define C_norm 0.35355339059327376220042218105242f

inline void
dct8(float *in, float *out){
    float X07P = in[0] + in[7];
    float X16P = in[1] + in[6];
    float X25P = in[2] + in[5];
    float X34P = in[3] + in[4];

    float X07M = in[0] - in[7];
    float X61M = in[6] - in[1];
    float X25M = in[2] - in[5];
    float X43M = in[4] - in[3];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    out[0] = C_norm * (X07P34PP + X16P25PP);
    out[2] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    out[4] = C_norm * (X07P34PP - X16P25PP);
    out[6] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    out[1] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    out[3] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    out[5] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    out[7] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

// Separable filter approach, I think.
// This kernel doesn't work. Will have to fix it sometime.
// Image dimensions must be divisible by 8.
__attribute__((uses_global_work_offset(0)))
__attribute__((reqd_work_group_size(1, 8, 1)))
__kernel
void
dct8x8(
    __global float * restrict src,
    __global float * restrict dst,
    uint height, uint width
) {
    const uint local_x = get_local_id(1);
    const uint global_y = get_group_id(0) * BLOCK_SIZE;
    const uint global_x = get_group_id(1) * BLOCK_SIZE + local_x;

    __local float transp[BLOCK_SIZE][BLOCK_SIZE];

    src += global_y * width + global_x;
    dst += global_y * width + global_x;

    float D_0[BLOCK_SIZE];
    float D_1[BLOCK_SIZE];
    float D_2[BLOCK_SIZE];
    float D_3[BLOCK_SIZE];

    // Write a column to transp.
    for(uint i = 0; i < BLOCK_SIZE; i++) {
        transp[i][local_x] = src[i * width];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = 0; i < BLOCK_SIZE; i++) {
        D_0[i] = transp[local_x][i];
    }
    dct8((__private float *)&D_0, (__private float *)&D_1);

    for(uint i = 0; i < BLOCK_SIZE; i++) {
        transp[local_x][i] = D_1[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = 0; i < BLOCK_SIZE; i++) {
        D_2[i] = transp[i][local_x];
    }
    dct8((__private float *)&D_2, (__private float *)&D_3);


    for(uint i = 0; i < BLOCK_SIZE; i++) {
        dst[i * width] = D_3[i];
    }
}

inline void
transpose(float buf[BLOCK_SIZE][BLOCK_SIZE]) {
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        for (int j = i + 1; j < BLOCK_SIZE; j++) {
            float t = buf[i][j];
            buf[i][j] = buf[j][i];
            buf[j][i] = t;
        }
    }
}

// This kernel should run fast on fpgas.
__attribute__((uses_global_work_offset(0)))
__kernel void
dct8x8_sd(
    __global float * restrict src,
    __global float * restrict dst,
    uint height, uint width
) {
    float buf0[BLOCK_SIZE][BLOCK_SIZE];
    float buf1[BLOCK_SIZE][BLOCK_SIZE];
    uint y0 = BLOCK_SIZE * get_global_id(0);
    uint x0 = BLOCK_SIZE * get_global_id(1);
    for (uint y1 = 0; y1 < BLOCK_SIZE; y1++) {
        for (uint x1 = 0; x1 < BLOCK_SIZE; x1++) {
            uint src_addr = (y0 + y1) * width + x0 + x1;
            buf0[y1][x1] = src[src_addr];
        }
    }
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        dct8((float *)&buf0[i], (float *)&buf1[i]);
    }
    transpose(buf1);
    for (uint i = 0; i < BLOCK_SIZE; i++) {
        dct8((float *)&buf1[i], (float *)&buf0[i]);
    }
    transpose(buf0);
    for (uint y1 = 0; y1 < BLOCK_SIZE; y1++) {
        for (uint x1 = 0; x1 < BLOCK_SIZE; x1++) {
            uint dst_addr = (y0 + y1) * width + x0 + x1;
            uint buf_addr = y1 * BLOCK_SIZE + x1;
            dst[dst_addr] = buf0[y1][x1];
        }
    }
}
