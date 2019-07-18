// Copyright (C) 2019 Björn Lindqvist
// SIMD example code
#include <assert.h>
#include <stdio.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

static void
test_sqrt() {
    __m128 v1 = _mm_set_ps(1.0, 4.0, 9.0, 16.0);
    v1 = _mm_sqrt_ps(v1);
    __m128 v2 = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    __m128 cmp = _mm_cmpeq_ps(v1, v2);
    assert(_mm_movemask_ps(cmp) == 0xf);
}

static void
test_max() {
    __m128 v1 = _mm_set_ps(3.0, 4.0, 10.0, -10.0);
    __m128 v2 = _mm_set_ps(-3.0, 8.0, 9.0, 0.0);
    __m128 v3 = _mm_max_ps(v1, v2);

    __m128 r = _mm_set_ps(3.0, 8.0, 10.0, 0.0);
    __m128 cmp = _mm_cmpeq_ps(v3, r);
    assert(_mm_movemask_ps(cmp) == 0xf);
}

// According to
// https://stackoverflow.com/questions/4120681/how-to-calculate-vector-dot-product-using-sse-intrinsic-functions-in-c
// this is a fast way to calculate the dot product using SIMD.
static void
test_dot_prod() {
    __m128 x = _mm_set_ps(2.0, 2.0, 1.0, 1.0);
    __m128 y = _mm_set_ps(10.0, 0.0, 0.0, -2.0);
    __m128 mr = _mm_mul_ps(x, y);
    __m128 shuf = _mm_movehdup_ps(mr);
    __m128 sums = _mm_add_ps(mr, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    float sum = _mm_cvtss_f32(sums);
    assert(sum == 18.0);
}

int
main(int argc, char *argv[]) {
    test_sqrt();
    test_max();
    test_dot_prod();
    return 0;
}
