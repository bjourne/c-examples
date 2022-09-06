// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// A separate file for the avx256 code for convenience.
#include <assert.h>
#include <immintrin.h>

#include "tensors/tensors-dct.h"

// Reduces typing
#define BCAST   _mm256_broadcast_ss
#define MUL     _mm256_mul_ps
#define ADD     _mm256_add_ps
#define SUB     _mm256_sub_ps
#define LOAD    _mm256_load_ps
#define LOADU   _mm256_loadu_ps
#define STORE   _mm256_store_ps
#define STOREU  _mm256_storeu_ps

static void
transpose(__m256 src[8], __m256 dst[8]) {
    dst[0] = _mm256_unpacklo_ps(src[0], src[1]);
    dst[1] = _mm256_unpackhi_ps(src[0], src[1]);
    dst[2] = _mm256_unpacklo_ps(src[2], src[3]);
    dst[3] = _mm256_unpackhi_ps(src[2], src[3]);
    dst[4] = _mm256_unpacklo_ps(src[4], src[5]);
    dst[5] = _mm256_unpackhi_ps(src[4], src[5]);
    dst[6] = _mm256_unpacklo_ps(src[6], src[7]);
    dst[7] = _mm256_unpackhi_ps(src[6], src[7]);

    src[0] = _mm256_shuffle_ps(dst[0], dst[2], _MM_SHUFFLE(1, 0, 1, 0));
    src[1] = _mm256_shuffle_ps(dst[0], dst[2], _MM_SHUFFLE(3, 2, 3, 2));
    src[2] = _mm256_shuffle_ps(dst[1], dst[3], _MM_SHUFFLE(1, 0, 1, 0));
    src[3] = _mm256_shuffle_ps(dst[1], dst[3], _MM_SHUFFLE(3, 2, 3, 2));
    src[4] = _mm256_shuffle_ps(dst[4], dst[6], _MM_SHUFFLE(1, 0, 1, 0));
    src[5] = _mm256_shuffle_ps(dst[4], dst[6], _MM_SHUFFLE(3, 2, 3, 2));
    src[6] = _mm256_shuffle_ps(dst[5], dst[7], _MM_SHUFFLE(1, 0, 1, 0));
    src[7] = _mm256_shuffle_ps(dst[5], dst[7], _MM_SHUFFLE(3, 2, 3, 2));

    dst[0] = _mm256_permute2f128_ps(src[0], src[4], 0x20);
    dst[1] = _mm256_permute2f128_ps(src[1], src[5], 0x20);
    dst[2] = _mm256_permute2f128_ps(src[2], src[6], 0x20);
    dst[3] = _mm256_permute2f128_ps(src[3], src[7], 0x20);
    dst[4] = _mm256_permute2f128_ps(src[0], src[4], 0x31);
    dst[5] = _mm256_permute2f128_ps(src[1], src[5], 0x31);
    dst[6] = _mm256_permute2f128_ps(src[2], src[6], 0x31);
    dst[7] = _mm256_permute2f128_ps(src[3], src[7], 0x31);
}

static void
nvidia_block(float * restrict src,
                    float * restrict ys,
                    int stride,
                    __m256 norm,
                    __m256 ca, __m256 cb, __m256 cc,
                    __m256 cd, __m256 ce, __m256 cf) {
    __m256 xa[8], ya[8];
    float xs[64] __attribute__ ((aligned (32)));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            xs[8 * i + j] = src[stride * j + i];
        }
    }
    for (int i = 0; i < 8; i++) {
        xa[i] = LOAD(&xs[8 * i]);
    }
    __m256 s10 = ADD(xa[0], xa[7]);
    __m256 s11 = ADD(xa[1], xa[6]);
    __m256 s12 = ADD(xa[2], xa[5]);
    __m256 s13 = ADD(xa[3], xa[4]);
    __m256 s14 = SUB(xa[0], xa[7]);
    __m256 s15 = SUB(xa[2], xa[5]);
    __m256 s16 = SUB(xa[4], xa[3]);
    __m256 s17 = SUB(xa[6], xa[1]);

    __m256 s20 = ADD(s10, s13);
    __m256 s21 = SUB(s10, s13);
    __m256 s22 = ADD(s11, s12);
    __m256 s23 = SUB(s11, s12);

    ya[0] = ADD(s20, s22);
    ya[1] = SUB(ADD(SUB(MUL(ca, s14), MUL(cc, s17)), MUL(cd, s15)), MUL(cf, s16));
    ya[2] = ADD(MUL(cb, s21), MUL(ce, s23));
    ya[3] = ADD(SUB(ADD(MUL(cc, s14), MUL(cf, s17)), MUL(ca, s15)), MUL(cd, s16));
    ya[4] = SUB(s20, s22);
    ya[5] = SUB(ADD(ADD(MUL(cd, s14), MUL(ca, s17)), MUL(cf, s15)), MUL(cc, s16));
    ya[6] = SUB(MUL(ce, s21), MUL(cb, s23));
    ya[7] = ADD(ADD(ADD(MUL(cf, s14), MUL(cd, s17)), MUL(cc, s15)), MUL(ca, s16));

    ya[0] = MUL(ya[0], norm);
    ya[1] = MUL(ya[1], norm);
    ya[2] = MUL(ya[2], norm);
    ya[3] = MUL(ya[3], norm);
    ya[4] = MUL(ya[4], norm);
    ya[5] = MUL(ya[5], norm);
    ya[6] = MUL(ya[6], norm);
    ya[7] = MUL(ya[7], norm);

    transpose(ya, xa);

    s10 = ADD(xa[0], xa[7]);
    s11 = ADD(xa[1], xa[6]);
    s12 = ADD(xa[2], xa[5]);
    s13 = ADD(xa[3], xa[4]);
    s14 = SUB(xa[0], xa[7]);
    s15 = SUB(xa[2], xa[5]);
    s16 = SUB(xa[4], xa[3]);
    s17 = SUB(xa[6], xa[1]);

    s20 = ADD(s10, s13);
    s21 = SUB(s10, s13);
    s22 = ADD(s11, s12);
    s23 = SUB(s11, s12);

    ya[0] = ADD(s20, s22);
    ya[1] = SUB(ADD(SUB(MUL(ca, s14), MUL(cc, s17)), MUL(cd, s15)), MUL(cf, s16));
    ya[2] = ADD(MUL(cb, s21), MUL(ce, s23));
    ya[3] = ADD(SUB(ADD(MUL(cc, s14), MUL(cf, s17)), MUL(ca, s15)), MUL(cd, s16));
    ya[4] = SUB(s20, s22);
    ya[5] = SUB(ADD(ADD(MUL(cd, s14), MUL(ca, s17)), MUL(cf, s15)), MUL(cc, s16));
    ya[6] = SUB(MUL(ce, s21), MUL(cb, s23));
    ya[7] = ADD(ADD(ADD(MUL(cf, s14), MUL(cd, s17)), MUL(cc, s15)), MUL(ca, s16));

    ya[0] = MUL(ya[0], norm);
    ya[1] = MUL(ya[1], norm);
    ya[2] = MUL(ya[2], norm);
    ya[3] = MUL(ya[3], norm);
    ya[4] = MUL(ya[4], norm);
    ya[5] = MUL(ya[5], norm);
    ya[6] = MUL(ya[6], norm);
    ya[7] = MUL(ya[7], norm);

    STORE(&ys[ 0], ya[0]);
    STORE(&ys[ 8], ya[1]);
    STORE(&ys[16], ya[2]);
    STORE(&ys[24], ya[3]);
    STORE(&ys[32], ya[4]);
    STORE(&ys[40], ya[5]);
    STORE(&ys[48], ya[6]);
    STORE(&ys[56], ya[7]);
}

void
tensor_dct8x8_nvidia_avx256_impl(float * restrict src, float * restrict dst,
                                 int width, int height) {
    assert(width % 8 == 0);
    assert(height % 8 == 0);

    const float c_norm = TENSOR_DCT8_NVIDIA_NORM;
    const float c_ca = TENSOR_DCT8_NVIDIA_CA;
    const float c_cb = TENSOR_DCT8_NVIDIA_CB;
    const float c_cc = TENSOR_DCT8_NVIDIA_CC;
    const float c_cd = TENSOR_DCT8_NVIDIA_CD;
    const float c_ce = TENSOR_DCT8_NVIDIA_CE;
    const float c_cf = TENSOR_DCT8_NVIDIA_CF;

    const __m256 norm = BCAST(&c_norm);
    const __m256 ca = BCAST(&c_ca);
    const __m256 cb = BCAST(&c_cb);
    const __m256 cc = BCAST(&c_cc);
    const __m256 cd = BCAST(&c_cd);
    const __m256 ce = BCAST(&c_ce);
    const __m256 cf = BCAST(&c_cf);


    float buf[64] __attribute__ ((aligned (32)));
    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {
            nvidia_block(&src[y * width + x], buf, width,
                         norm, ca, cb, cc, cd, ce, cf);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    dst[(y + i) * width + j + x] = buf[8 * i + j];
                }
            }
        }
    }


}
