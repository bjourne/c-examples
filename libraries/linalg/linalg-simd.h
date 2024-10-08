// Copyright (C) 2019, 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef LINALG_SIMD_H
#define LINALG_SIMD_H

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include "linalg/linalg.h"

// Naming conventions:
//
// * set_Nx: use N values to set all N vector elements.
// * set_1x: broadcast one value to all N vector elements.
// * nameX: register containing X values of type name.
// * set_Nx_cM: cast from one vector type to another using a
//   cvt-function.


typedef __m128 float4;
typedef __m128i int4;

// float4 functions
inline float4
f4_load(float const *ptr) {
    return _mm_load_ps(ptr);
}

inline void
f4_store(float4 r, float *ptr) {
    _mm_store_ps(ptr, r);
}

inline float4
f4_tern(float4 mask, float4 a, float4 b) {
    return _mm_blendv_ps(b, a, mask);
}

inline float4
f4_set_4x(float a, float b, float c, float d) {
    return _mm_set_ps(d, c, b, a);
}

inline float4
f4_set_1x(float a) {
    return _mm_set1_ps(a);
}

inline float4
f4_0() {
    return _mm_setzero_ps();
}

inline float4
f4_1() {
    return f4_set_1x(1);
}

inline float4
f4_2() {
    return f4_set_1x(2);
}

inline float4
f4_4() {
    return f4_set_1x(4);
}

inline float4
f4_set_4x_i(int32_t a, int32_t b, int32_t c, int32_t d) {
    return _mm_castsi128_ps(_mm_set_epi32(a, b, c, d));
}

inline uint32_t
f4_movemask(float4 a) {
    return _mm_movemask_ps(a);
}

inline float4
f4_mul(float4 a, float4 b) {
    return _mm_mul_ps(a, b);
}

inline float4
f4_div(float4 a, float4 b) {
    return _mm_div_ps(a, b);
}

inline float4
f4_cmp_lte(float4 a, float4 b) {
    return _mm_cmple_ps(a, b);
}

inline float4
f4_add(float4 a, float4 b) {
    return _mm_add_ps(a, b);
}

inline float4
f4_sub(float4 a, float4 b) {
    return _mm_sub_ps(a, b);
}

inline float4
f4_and(float4 a, float4 b) {
    return _mm_and_ps(a, b);
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
    float4 cmp = _mm_cmpeq_ps(a, b);
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
f4_fma(float4 a, float4 b, float4 c) {
#if defined(__AVX2__)
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

// double2 functions
typedef __m128d double2;

inline double2
d2_set_1x(double a) {
    return _mm_set1_pd(a);
}

inline double2
d2_0() {
    return _mm_setzero_pd();
}

inline double2
d2_load(double *ptr) {
    return _mm_load_pd(ptr);
}

inline void
d2_store(double2 a, double *ptr) {
    _mm_store_pd(ptr, a);
}

inline double2
d2_mul(double2 a, double2 b) {
    return _mm_mul_pd(a, b);
}

inline double2
d2_add(double2 a, double2 b) {
    return _mm_add_pd(a, b);
}

inline double2
d2_cmp_gte(double2 a, double2 b) {
    return _mm_cmp_pd(a, b, _CMP_GE_OQ);
}

inline uint32_t
d2_movemask(double2 a) {
    return _mm_movemask_pd(a);
}

inline double2
d2_tern(double2 mask, double2 a, double2 b) {
    return _mm_blendv_pd(b, a, mask);
}

inline double2
d2_andnot(double2 a, double2 b) {
    return _mm_andnot_pd(a, b);
}

// long2 functions
typedef __m128i long2;

inline long2
l2_load(int64_t *ptr) {
    return _mm_load_si128((const __m128i *)ptr);
}

inline void
l2_store(long2 a, int64_t *ptr) {
    _mm_store_si128((long2 *)ptr, a);
}

inline long2
l2_sub(long2 a, long2 b) {
    return _mm_sub_epi64(a, b);
}

inline long2
l2_and(long2 a, long2 b) {
    return _mm_and_si128(a, b);
}

inline long2
l2_set_1x(int64_t a) {
    return _mm_set1_epi64x(a);
}

inline long2
l2_1() {
    return l2_set_1x(1);
}

inline long2
l2_tern(long2 mask, long2 a, long2 b) {
    return (long2)d2_tern((double2)mask, (double2)a, (double2)b);
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
    return _mm_setzero_si128();
}

inline int4
i4_1() {
    return _mm_set1_epi32(1);
}

inline int4
i4_sub(int4 a, int4 b) {
    return _mm_sub_epi32(a, b);
}

inline int4
i4_add(int4 a, int4 b) {
    return _mm_add_epi32(a, b);
}

inline int4
i4_and(int4 a, int4 b) {
    return _mm_and_si128(a, b);
}

inline bool
i4_all_eq(int4 a, int4 b) {
    float4 cmp = (float4)_mm_cmpeq_epi32(a, b);
    return _mm_movemask_ps(cmp) == 0xf;
}

inline int4
i4_test(int4 a) {
    return _mm_cmpeq_epi32(a, i4_0());
}

// Chooses a if mask is negative, else b.
inline int4
i4_tern(int4 mask, int4 a, int4 b) {
    return (int4)f4_tern((float4)mask, (float4)a, (float4)b);
}

inline void
i4_storeu(int4 a, int32_t *d) {
    _mm_storeu_si128((int4 *)d, a);
}

inline void
i4_store(int4 a, int32_t *d) {
    _mm_store_si128((int4 *)d, a);
}

inline void
i4_print(int4 r) {
    int32_t d[4];
    i4_storeu(r, d);
    // Note order
    printf("{%d, %d, %d, %d}", d[0], d[1], d[2], d[3]);
}

#ifdef __AVX2__

typedef __m256d double4;
typedef __m256i int8;
typedef __m256i long4;
typedef __m256 float8;

inline void
f8_store(float8 r, float *ptr) {
    _mm256_store_ps(ptr, r);
}

inline uint32_t
f8_is_zero(float8 a) {
    // This may not be more efficient than movemask.
    return _mm256_testz_si256((int8)a, (int8)a);
}

inline uint32_t
f8_movemask(float8 a) {
    return _mm256_movemask_ps(a);
}

inline float8
f8_set_8x(float a, float b, float c, float d,
          float e, float f, float g, float h) {
    return _mm256_set_ps(h, g, f, e, d, c, b, a);
}

inline float8
f8_set_1x(float a) {
    return _mm256_set1_ps(a);
}

inline float8
f8_set_8x_i8(int8 a) {
    return _mm256_cvtepi32_ps(a);
}

inline float8
f8_add(float8 a, float8 b) {
    return _mm256_add_ps(a, b);
}

inline float8
f8_sub(float8 a, float8 b) {
    return _mm256_sub_ps(a, b);
}

inline float8
f8_mul(float8 a, float8 b) {
    return _mm256_mul_ps(a, b);
}

inline float8
f8_div(float8 a, float8 b) {
    return _mm256_div_ps(a, b);
}

inline float8
f8_fma(float8 a, float8 b, float8 c) {
    return _mm256_fmadd_ps(a, b, c);
}

inline float8
f8_and(float8 a, float8 b) {
    return _mm256_and_ps(a, b);
}

inline float8
f8_0() {
    return _mm256_setzero_ps();
}

inline float8
f8_2() {
    return f8_set_1x(2);
}

inline float8
f8_1() {
    return f8_set_1x(1);
}

inline float8
f8_4() {
    return f8_set_1x(4);
}

inline float8
f8_cmp_lte(float8 a, float8 b) {
    return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}

// double4
inline uint32_t
d4_movemask(double4 a) {
    return _mm256_movemask_pd(a);
}

inline uint32_t
d4_count_mask(double4 a) {
    return _mm_popcnt_u32(d4_movemask(a));
}

inline double4
d4_andnot(double4 a, double4 b) {
    return _mm256_andnot_pd(a, b);
}

inline double4
d4_load(double *ptr) {
    return _mm256_load_pd(ptr);
}

inline double4
d4_set_1x(double a) {
    return _mm256_set1_pd(a);
}

inline double4
d4_set_4x_f4(float4 a) {
    return _mm256_cvtps_pd(a);
}

inline double4
d4_0() {
    return _mm256_setzero_pd();
}

inline double4
d4_mul(double4 a, double4 b) {
    return _mm256_mul_pd(a, b);
}

inline double4
d4_add(double4 a, double4 b) {
    return _mm256_add_pd(a, b);
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
d4_store(double4 r, double *ptr) {
    _mm256_store_pd(ptr, r);
}

inline void
d4_storeu(double4 r, double *ptr) {
    _mm256_storeu_pd(ptr, r);
}


inline void
d4_print(double4 a, int n_dec) {
    double r[4];
    d4_storeu(a, r);
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

// int8
inline int8
i8_set_1x(int32_t a) {
    return _mm256_set1_epi32(a);
}

inline int8
i8_0() {
    return _mm256_setzero_si256();
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

inline int8
i8_abs(int8 a) {
    return _mm256_abs_epi32(a);
}

inline int8
i8_sub(int8 a, int8 b) {
    return _mm256_sub_epi32(a, b);
}

// long4
inline long4
l4_tern(long4 mask, long4 a, long4 b) {
    return (long4)d4_tern((double4)mask, (double4)a, (double4)b);
}

inline long4
l4_load(int64_t *ptr) {
    return _mm256_load_si256((long4 *)ptr);
}

inline long4
l4_sub(long4 a, long4 b) {
    return _mm256_sub_epi64(a, b);
}

inline long4
l4_set_1x(int64_t a) {
    return _mm256_set1_epi64x(a);
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

inline long4
l4_1() {
    return l4_set_1x(1);
}

inline long4
l4_and(long4 a, long4 b) {
    return _mm256_and_si256(a, b);
}

inline void
l4_store(long4 a, int64_t *d) {
    _mm256_store_si256((long4 *)d, a);
}

inline int4
l4_cvt_i4(long4 r) {
    // Emulates _mm256_cvtepi64_epi32
    int8 pat = i8_set_8x(0, 2, 4, 6, 0, 0, 0, 0);
    return _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(r, pat));
}

inline long4
i4_cvt_l4(int4 r) {
    return _mm256_cvtepi32_epi64(r);
}

inline double4
i4_cvt_d4(int4 r) {
    return (double4)i4_cvt_l4(r);
}

inline int4
d4_cvt_i4(double4 r) {
    return l4_cvt_i4((long4)r);
}

#endif

#ifdef __AVX512F__

typedef __m512 float16;
typedef __m512i int16;

inline float16
f16_set_16x(
    float a, float b, float c, float d,
    float e, float f, float g, float h,
    float i, float j, float k, float l,
    float m, float n, float o, float p
) {
    return _mm512_set_ps(
        p, o, n, m, l, k, j, i,
        h, g, f, e, d, c, b, a
    );
}

inline float16
f16_set_1x(float a) {
    return _mm512_set1_ps(a);
}

inline float16
f16_add(float16 a, float16 b) {
    return _mm512_add_ps(a, b);
}

inline float16
f16_sub(float16 a, float16 b) {
    return _mm512_sub_ps(a, b);
}

inline float16
f16_mul(float16 a, float16 b) {
    return _mm512_mul_ps(a, b);
}

inline float16
f16_div(float16 a, float16 b) {
    return _mm512_div_ps(a, b);
}

inline float16
f16_0() {
    return _mm512_setzero_ps();
}

inline float16
f16_4() {
    return f16_set_1x(4);
}

inline __mmask16
f16_cmp_lte(float16 a, float16 b) {
    return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
}

inline float16
f16_fma(float16 a, float16 b, float16 c) {
    return _mm512_fmadd_ps(a, b, c);
}

inline float16
f16_set_16x_i16(int16 a) {
    return _mm512_cvtepi32_ps(a);
}

inline void
f16_store(float16 r, float *ptr) {
    _mm512_store_ps(ptr, r);
}

inline int16
i16_set_1x(int a) {
    return _mm512_set1_epi32(a);
}

inline int16
i16_0() {
    return _mm512_setzero_si512();
}

inline int16
i16_1() {
    return i16_set_1x(1);
}

inline int16
i16_add_masked(int16 a, int16 b, __mmask16 mask) {
    return _mm512_mask_add_epi32(a, mask, a, b);
}

inline void
i16_print(int16 r) {
    int32_t d[16];
    _mm512_storeu_si512((__m256i *)d, r);
    printf("{");
    for (uint32_t i = 0; i < 16; i++) {
        printf("%3d", d[i]);
        if (i < 15) {
            printf(", ");
        }
    }
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
    return f4_fma(a.x, b.x, f4_fma(a.y, b.y, _mm_mul_ps(a.z, b.z)));
}


#endif
