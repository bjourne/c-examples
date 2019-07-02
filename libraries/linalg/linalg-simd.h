// Copyright (C) 2017 Björn Lindqvist
#ifndef LINALG_SIMD_H
#define LINALG_SIMD_H

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

#include "linalg/linalg.h"

// vec3x4 type. Four 3d vectors in packed format to exploit SIMD.
typedef struct {
    __m128 x;
    __m128 y;
    __m128 z;
} vec3x4;

inline vec3x4
v3x4_from_vecs(vec3 vecs[4]) {
    vec3x4 ret;
    for (int i = 0; i < 4; i++) {
        ret.x[i] = vecs[i].x;
        ret.y[i] = vecs[i].y;
        ret.z[i] = vecs[i].z;
    }
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
inline __m128
v3x4_dot(vec3x4 a, vec3x4 b) {
    return _mm_fmadd_ps(a.x, b.x, _mm_fmadd_ps(a.y, b.y, _mm_mul_ps(a.z, b.z)));
}





#endif
