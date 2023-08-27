// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef LINALG_SIMD_H
#define LINALG_SIMD_H

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

#include "linalg/linalg.h"

typedef __m128 float4;
typedef __m128i int4;

// float4 functions
inline float4
f4_set_4x_i(int32_t a, int32_t b, int32_t c, int32_t d) {
    return _mm_castsi128_ps(_mm_set_epi32(a, b, c, d));
}

inline float4
f4_abs(float4 a) {
    float4 signmask =
        _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_andnot_ps(signmask, a);
}

inline float4
f4_signmask(float4 a) {
    float4 signmask =
        _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_and_ps(a, signmask);
}

inline bool
f4_all_eq(float4 a, float4 b) {
    __m128 cmp = _mm_cmpeq_ps(a, b);
    return _mm_movemask_ps(cmp) == 0xf;
}

inline float4
f4_scale(float4 a, float s) {
    return _mm_mul_ps(a, _mm_set1_ps(s));
}

// Check if this can be improved.
inline float4
f4_broadcast(float4 a, int i) {
    union { float4 reg; float f[4]; } r = { .reg = a };
    return _mm_set1_ps(r.f[i]);
}

inline void
f4_print(float4 a, int n_dec) {
    float r[4];
    _mm_storeu_ps(r, a);
    printf("{");
    la_print_float(r[3], n_dec);
    printf(", ");
    la_print_float(r[2], n_dec);
    printf(", ");
    la_print_float(r[1], n_dec);
    printf(", ");
    la_print_float(r[0], n_dec);
    printf("}");
}

inline float4
madd(float4 a, float4 b, float4 c) {
#if defined(__AVX2__)
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

// int4 functions
inline int4
i4_load(int32_t *ptr) {
    return _mm_load_si128((const __m128i *)ptr);
}

inline int4
i4_set_4x(int32_t a, int32_t b, int32_t c, int32_t d) {
    // Note order
    return _mm_set_epi32(d, c, b, a);
}

inline int4
i4_set_1x(int32_t a) {
    return _mm_set1_epi32(a);
}

inline int4
i4_0() {
    return _mm_set1_epi32(0);
}

inline int4
i4_1() {
    return _mm_set1_epi32(1);
}

inline int4
i4_sub(int4 a, int4 b) {
    return _mm_sub_epi32(a, b);
}

inline bool
i4_all_eq(int4 a, int4 b) {
    __m128 cmp = _mm_cmpeq_epi32(a, b);
    return _mm_movemask_ps(cmp) == 0xf;
}

inline int4
i4_test(int4 a) {
    return _mm_cmpeq_epi32(a, i4_set_1x(0));
}

// Chooses a if mask is negative, else b.
inline int4
i4_tern(int4 mask, int4 a, int4 b) {
    return _mm_blendv_ps(b, a, mask);
}

inline void
i4_print(int4 r) {
    int32_t d[4];
    _mm_storeu_si128((__m128i *)d, r);
    // Note order
    printf("{%d, %d, %d, %d}", d[0], d[1], d[2], d[3]);
}

#ifdef __AVX2__

typedef __m256d double4;
typedef __m256i int8;
typedef __m256i long4;

inline double4
d4_set_1x(double a) {
    return _mm256_set1_pd(a);
}

inline double4
d4_0() {
    return d4_set_1x(0);
}

inline double4
d4_set_4x(double a, double b, double c, double d) {
    // Note order
    return _mm256_set_pd(d, c, b, a);
}

inline double4
d4_set_4x_i(int32_t a, int32_t b, int32_t c, int32_t d) {
    return _mm256_set_pd((double)d, (double)c, (double)b, (double)a);
}

inline double4
d4_cmp_gte(double4 a, double4 b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}

inline double4
d4_tern(double4 mask, double4 a, double4 b) {
    return _mm256_blendv_pd(b, a, mask);
}

inline void
d4_print(double4 a, int n_dec) {
    double r[4];
    _mm256_storeu_pd(r, a);
    printf("{");
    la_print_float(r[0], n_dec);
    printf(", ");
    la_print_float(r[1], n_dec);
    printf(", ");
    la_print_float(r[2], n_dec);
    printf(", ");
    la_print_float(r[3], n_dec);
    printf("}");
}

inline long4
l4_set_4x(int64_t a, int64_t b, int64_t c, int64_t d) {
    // Note order
    return _mm256_set_epi64x(d, c, b, a);
}

inline void
l4_print(long4 r) {
    int64_t d[4];
    _mm256_storeu_si256((__m256i *)d, r);
    printf("{%ld, %ld, %ld, %ld}", d[0], d[1], d[2], d[3]);
}


inline int8
i8_set_8x(int32_t a, int32_t b, int32_t c, int32_t d,
          int32_t e, int32_t f, int32_t g, int32_t h) {
    return _mm256_set_epi32(h, g, f, e, d, c, b, a);
}

inline void
i8_print(int8 r) {
    int32_t d[8];
    _mm256_storeu_si256((__m256i *)d, r);
    printf("{%d, %d, %d, %d, %d, %d, %d, %d}",
           d[0], d[1], d[2], d[3],
           d[4], d[5], d[6], d[7]);
}

inline int4
l4_cvt_i4(long4 r) {
    // Emulates _mm256_cvtepi64_epi32
    int8 pat = i8_set_8x(0, 2, 4, 6, 0, 0, 0, 0);
    return _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(r, pat));
}

#endif

// vec3x4 type. Four 3d vectors in packed format to exploit SIMD.
typedef struct {
    float4 x;
    float4 y;
    float4 z;
} vec3x4;

inline vec3x4
v3x4_from_vecs(vec3 vecs[4]) {
    float x[4], y[4], z[4];
    for (int i = 0; i < 4; i++) {
        x[i] = vecs[i].x;
        y[i] = vecs[i].y;
        z[i] = vecs[i].z;
    }
    vec3x4 ret;
    ret.x = _mm_set_ps(x[0], x[1], x[2], x[3]);
    ret.y = _mm_set_ps(y[0], y[1], y[2], y[3]);
    ret.z = _mm_set_ps(z[0], z[1], z[2], z[3]);
    return ret;
}

// Binary operators
inline vec3x4
v3x4_add(vec3x4 a, vec3x4 b) {
    return (vec3x4){
        _mm_add_ps(a.x, b.x),
        _mm_add_ps(a.y, b.y),
        _mm_add_ps(a.z, b.z)
    };
}
inline vec3x4
v3x4_mul(vec3x4 a, vec3x4 b) {
    return (vec3x4){
        _mm_mul_ps(a.x, b.x),
        _mm_mul_ps(a.y, b.y),
        _mm_mul_ps(a.z, b.z)
    };
}

// Dot product
inline float4
v3x4_dot(vec3x4 a, vec3x4 b) {
    return madd(a.x, b.x, madd(a.y, b.y, _mm_mul_ps(a.z, b.z)));
}


#endif
