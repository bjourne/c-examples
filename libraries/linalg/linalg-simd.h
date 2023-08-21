// Copyright (C) 2019, 2023 Björn Lindqvist <bjourne@gmail.com>
#ifndef LINALG_SIMD_H
#define LINALG_SIMD_H

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

#include "linalg/linalg.h"

typedef __m128 float4;

inline float4
f4_set_int32(int32_t a, int32_t b, int32_t c, int32_t d) {
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
f4_eq(float4 a, float4 b) {
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

#ifdef __AVX2__
typedef __m256d double4;

inline double4
d4_set_int32(int32_t a, int32_t b, int32_t c, int32_t d) {
    return _mm256_set_pd((double)a, (double)b, (double)c, (double)d);
}

inline void
d4_print(double4 a, int n_dec) {
    double r[4];
    _mm256_storeu_pd(r, a);
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
