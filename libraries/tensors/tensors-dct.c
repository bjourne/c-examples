// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensors-dct.h"

static float SQRT2 = 1.4142135623730951f;
static float SQRT2INV = 0.7071067811865475f;

extern inline void tensor_dct8_loeffler(float x[8], float y[8]);
extern inline void tensor_dct8_nvidia(float * restrict x,
                                      float * restrict y);

void
tensor_dct2d_rect(tensor *src, tensor *dst,
                  int sy, int sx,
                  int height, int width) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int n_cols = src->dims[1];
    float pi_div_2rows = 0.5 * M_PI / height;
    float pi_div_2cols = 0.5 * M_PI / width;
    float coeff = 2.0 / sqrt(height * width);

    for (int u = 0; u < height; u++) {
        for (int v = 0; v < width; v++) {
            float o = 0.0f;
            for (int y = 0; y < height;  y++) {
                for (int x = 0; x < width; x++) {
                    float cos_y = cos((2 * y + 1) * u * pi_div_2rows);
                    float cos_x = cos((2 * x + 1) * v * pi_div_2cols);
                    int r_addr = n_cols * (y + sy) + x + sx;
                    o  += src->data[r_addr] * cos_y * cos_x;
                }
            }
            float c_u = u == 0 ? SQRT2INV : 1;
            float c_v = v == 0 ? SQRT2INV : 1;
            int w_addr = n_cols * (u + sy) + v + sx;
            dst->data[w_addr] = coeff * c_u * c_v * o;
        }
    }
}

void
tensor_dct2d_blocks(tensor *src, tensor *dst,
                    int block_height, int block_width) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int height = src->dims[0];
    int width = src->dims[1];

    // Lazyness
    assert(width % block_width == 0);
    assert(height % block_height == 0);

    for (int y = 0; y < height; y += block_height) {
        for (int x = 0; x < width; x += block_width) {
            tensor_dct2d_rect(src, dst, y, x, block_height, block_width);
        }
    }
}

// Reduces typing
#define BCAST   _mm256_broadcast_ss
#define MUL     _mm256_mul_ps
#define ADD     _mm256_add_ps
#define SUB     _mm256_sub_ps
#define LOAD    _mm256_load_ps
#define STORE   _mm256_store_ps

static void
tensor_dct8_nvidia_avx256(float * restrict x, float * restrict y) {
    const float c_norm
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_NORM;
    const float c_ca
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_CA;
    const float c_cb
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_CB;
    const float c_cc
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_CC;
    const float c_cd
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_CD;
    const float c_ce
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_CE;
    const float c_cf
        __attribute__ ((aligned (16))) = TENSOR_DCT8_NVIDIA_CF;

    const __m256 norm = BCAST(&c_norm);
    const __m256 ca = BCAST(&c_ca);
    const __m256 cb = BCAST(&c_cb);
    const __m256 cc = BCAST(&c_cc);
    const __m256 cd = BCAST(&c_cd);
    const __m256 ce = BCAST(&c_ce);
    const __m256 cf = BCAST(&c_cf);

    float xs[64] __attribute__ ((aligned (32)));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            xs[8 * i + j] = x[8 * j + i];
        }
    }
    __m256 x0 = LOAD(&xs[0]);
    __m256 x1 = LOAD(&xs[8]);
    __m256 x2 = LOAD(&xs[16]);
    __m256 x3 = LOAD(&xs[24]);
    __m256 x4 = LOAD(&xs[32]);
    __m256 x5 = LOAD(&xs[40]);
    __m256 x6 = LOAD(&xs[48]);
    __m256 x7 = LOAD(&xs[56]);

    __m256 s10 = ADD(x0, x7);
    __m256 s11 = ADD(x1, x6);
    __m256 s12 = ADD(x2, x5);
    __m256 s13 = ADD(x3, x4);
    __m256 s14 = SUB(x0, x7);
    __m256 s15 = SUB(x2, x5);
    __m256 s16 = SUB(x4, x3);
    __m256 s17 = SUB(x6, x1);

    __m256 s20 = ADD(s10, s13);
    __m256 s21 = SUB(s10, s13);
    __m256 s22 = ADD(s11, s12);
    __m256 s23 = SUB(s11, s12);

    __m256 y0 = ADD(s20, s22);
    __m256 y1 = SUB(ADD(SUB(MUL(ca, s14), MUL(cc, s17)), MUL(cd, s15)), MUL(cf, s16));
    __m256 y2 = ADD(MUL(cb, s21), MUL(ce, s23));
    __m256 y3 = ADD(SUB(ADD(MUL(cc, s14), MUL(cf, s17)), MUL(ca, s15)), MUL(cd, s16));
    __m256 y4 = SUB(s20, s22);
    __m256 y5 = SUB(ADD(ADD(MUL(cd, s14), MUL(ca, s17)), MUL(cf, s15)), MUL(cc, s16));
    __m256 y6 = SUB(MUL(ce, s21), MUL(cb, s23));
    __m256 y7 = ADD(ADD(ADD(MUL(cf, s14), MUL(cd, s17)), MUL(cc, s15)), MUL(ca, s16));

    y0 = MUL(y0, norm);
    y1 = MUL(y1, norm);
    y2 = MUL(y2, norm);
    y3 = MUL(y3, norm);
    y4 = MUL(y4, norm);
    y5 = MUL(y5, norm);
    y6 = MUL(y6, norm);
    y7 = MUL(y7, norm);

    x0 = _mm256_unpacklo_ps(y0, y1);
    x1 = _mm256_unpackhi_ps(y0, y1);
    x2 = _mm256_unpacklo_ps(y2, y3);
    x3 = _mm256_unpackhi_ps(y2, y3);
    x4 = _mm256_unpacklo_ps(y4, y5);
    x5 = _mm256_unpackhi_ps(y4, y5);
    x6 = _mm256_unpacklo_ps(y6, y7);
    x7 = _mm256_unpackhi_ps(y6, y7);

    y0 = _mm256_shuffle_ps(x0,x2,_MM_SHUFFLE(1,0,1,0));
    y1 = _mm256_shuffle_ps(x0,x2,_MM_SHUFFLE(3,2,3,2));
    y2 = _mm256_shuffle_ps(x1,x3,_MM_SHUFFLE(1,0,1,0));
    y3 = _mm256_shuffle_ps(x1,x3,_MM_SHUFFLE(3,2,3,2));
    y4 = _mm256_shuffle_ps(x4,x6,_MM_SHUFFLE(1,0,1,0));
    y5 = _mm256_shuffle_ps(x4,x6,_MM_SHUFFLE(3,2,3,2));
    y6 = _mm256_shuffle_ps(x5,x7,_MM_SHUFFLE(1,0,1,0));
    y7 = _mm256_shuffle_ps(x5,x7,_MM_SHUFFLE(3,2,3,2));

    x0 = _mm256_permute2f128_ps(y0, y4, 0x20);
    x1 = _mm256_permute2f128_ps(y1, y5, 0x20);
    x2 = _mm256_permute2f128_ps(y2, y6, 0x20);
    x3 = _mm256_permute2f128_ps(y3, y7, 0x20);
    x4 = _mm256_permute2f128_ps(y0, y4, 0x31);
    x5 = _mm256_permute2f128_ps(y1, y5, 0x31);
    x6 = _mm256_permute2f128_ps(y2, y6, 0x31);
    x7 = _mm256_permute2f128_ps(y3, y7, 0x31);

    s10 = ADD(x0, x7);
    s11 = ADD(x1, x6);
    s12 = ADD(x2, x5);
    s13 = ADD(x3, x4);
    s14 = SUB(x0, x7);
    s15 = SUB(x2, x5);
    s16 = SUB(x4, x3);
    s17 = SUB(x6, x1);

    s20 = ADD(s10, s13);
    s21 = SUB(s10, s13);
    s22 = ADD(s11, s12);
    s23 = SUB(s11, s12);

    y0 = ADD(s20, s22);
    y1 = SUB(ADD(SUB(MUL(ca, s14), MUL(cc, s17)), MUL(cd, s15)), MUL(cf, s16));
    y2 = ADD(MUL(cb, s21), MUL(ce, s23));
    y3 = ADD(SUB(ADD(MUL(cc, s14), MUL(cf, s17)), MUL(ca, s15)), MUL(cd, s16));
    y4 = SUB(s20, s22);
    y5 = SUB(ADD(ADD(MUL(cd, s14), MUL(ca, s17)), MUL(cf, s15)), MUL(cc, s16));
    y6 = SUB(MUL(ce, s21), MUL(cb, s23));
    y7 = ADD(ADD(ADD(MUL(cf, s14), MUL(cd, s17)), MUL(cc, s15)), MUL(ca, s16));

    y0 = MUL(y0, norm);
    y1 = MUL(y1, norm);
    y2 = MUL(y2, norm);
    y3 = MUL(y3, norm);
    y4 = MUL(y4, norm);
    y5 = MUL(y5, norm);
    y6 = MUL(y6, norm);
    y7 = MUL(y7, norm);

    STORE(&y[0], y0);
    STORE(&y[8], y1);
    STORE(&y[16], y2);
    STORE(&y[24], y3);
    STORE(&y[32], y4);
    STORE(&y[40], y5);
    STORE(&y[48], y6);
    STORE(&y[56], y7);
}




// 6.28
static void
dct8x8_nvidia(float * restrict src, float * restrict dst,
              int stride) {
    float x[64] __attribute__ ((aligned (32)));
    float y[64] __attribute__ ((aligned (32)));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            x[8 * i + j] = src[stride * i + j];
        }
    }
    tensor_dct8_nvidia_avx256(x, y);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            dst[stride * i + j] = y[8 * i + j];
        }
    }


}

void
tensor_dct2d_8x8_blocks_nvidia(tensor *src, tensor *dst) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int height = src->dims[0];
    int width = src->dims[1];


    // Lazyness
    assert(width % 8 == 0);
    assert(height % 8 == 0);

    float *src_data = src->data;
    float *dst_data = dst->data;
    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {
            dct8x8_nvidia(&src_data[y * width + x],
                          &dst_data[y * width + x],
                          width);
        }
    }
}


void
tensor_idct2d(tensor *src, tensor *dst) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int n_rows = src->dims[0];
    int n_cols = src->dims[1];

    float pi_div_2rows = 0.5 * M_PI / n_rows;
    float pi_div_2cols = 0.5 * M_PI / n_cols;
    float denom = sqrt(n_rows * n_cols);
    for  (unsigned int u = 0; u < n_rows; u++) {
        for (unsigned int v = 0; v < n_cols; v++) {
            float o = 0.0f;
            for (unsigned int y = 0; y < n_rows; y++) {
                for (unsigned int x = 0; x < n_cols; x++) {
                    float a_y = y == 0 ? 1.0 : SQRT2;
                    float a_x = x == 0 ? 1.0 : SQRT2;
                    float cos_y = cos((2 * u + 1) * y * pi_div_2rows);
                    float cos_x = cos((2 * v + 1) * x * pi_div_2cols);
                    o += src->data[n_cols * y + x] * a_y * a_x * cos_y * cos_x / denom;
                }
            }
            dst->data[n_cols * u + v] = o;
        }
    }
}
