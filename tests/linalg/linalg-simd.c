// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"
#include "linalg/linalg-simd.h"

// f4 tests
void
test_f4_abs() {
    float4 a = _mm_set_ps(-7.8f, 3.2f, 0.0f, -1.2f);
    float4 r = _mm_set_ps(7.8f, 3.2f, 0.0f, 1.2f);
    assert(f4_all_eq(f4_abs(a), r));
}

void
test_f4_signmask() {
    float4 a = _mm_set_ps(-7.8f, 3.2f, 0.0f, -1.2f);
    float4 b = f4_signmask(a);
    float4 r = f4_set_4x_i(0x80000000, 0, 0, 0x80000000);
    assert(f4_all_eq(b, r));
}

// v3x4 tests
void
test_from_vecs() {
    vec3x4 v = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    float4 x4 = _mm_set_ps(1, 4, 7, 10);
    float4 y4 = _mm_set_ps(2, 5, 8, 11);
    float4 z4 = _mm_set_ps(3, 6, 9, 12);
    assert(f4_all_eq(v.x, x4));
    assert(f4_all_eq(v.y, y4));
    assert(f4_all_eq(v.z, z4));
}

void
test_add() {
    vec3x4 a = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    vec3x4 b = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    vec3x4 c = v3x4_add(a, b);
    float4 x4 = _mm_set_ps(2, 8, 14, 20);
    assert(f4_all_eq(c.x, x4));
}

void
test_dot() {
    vec3x4 a = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    vec3x4 b = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    float4 dp = v3x4_dot(a, b);
    float4 r = _mm_set_ps(14, 77, 194, 365);
    assert(f4_all_eq(dp, r));
}

void
test_broadcast() {
    float4 a = _mm_set_ps(3, 4, 5, 6);
    assert(f4_all_eq(f4_broadcast(a, 3), _mm_set_ps(3, 3, 3, 3)));
    assert(f4_all_eq(f4_broadcast(a, 2), _mm_set_ps(4, 4, 4, 4)));
}

void
test_f4_xor() {
    float4 a = f4_set_4x_i(255, 10, 0, 8);
    float4 b = f4_set_4x_i(3, 7, 0, 0x80);
    float4 c = _mm_xor_ps(a, b);
    float4 d = f4_set_4x_i(255 ^ 3, 0xa ^ 7, 0 ^ 0, 8 ^ 0x80);
    assert(f4_all_eq(c, d));
}

void
test_f4_or() {
    float4 a = f4_set_4x_i(5, 6, 7, 100);
    float4 b = f4_set_4x_i(3, 7, 0, 0x80);
    float4 c = _mm_or_ps(a, b);
    float4 d = f4_set_4x_i(5 | 3, 6 | 7, 7 | 0, 100 | 0x80);
    assert(f4_all_eq(c, d));
}

void
test_f4_scale() {
    float4 a = _mm_set_ps(0, 1, 2, 3);
    float4 b = _mm_set_ps(0, 6, 12, 18);
    assert(f4_all_eq(f4_scale(a, 6), b));
}

void
test_d4_set_i8() {
    double4 a = d4_set_4x_i(123, 456, 789, 222);
    d4_print(a, 3);
    printf("\n");
}

void
test_i4() {
    int32_t data[] = {1, 2, 3, 4};
    int4 reg1 = i4_load(data);
    int4 reg2 = i4_set_1x(1);
    int4 reg3 = i4_set_4x(0, 1, 2, 3);
    reg1 = i4_sub(reg1, reg2);
    assert(i4_all_eq(reg1, reg3));
}

void
test_i4_blend() {
    int4 event = i4_set_4x(0, 0, -1, 0);
    int4 cnt = i4_load((int32_t[]){3, 0, 0, 0});
    int4 cnt_next = i4_sub(cnt, i4_1());

    int4 reg3 = i4_tern(event, i4_set_1x(9), i4_0());
    int4 reg4 = i4_tern(cnt_next, reg3, cnt_next);
    assert(i4_all_eq(reg4, i4_set_4x(2, 0, 9, 0)));
}

void
test_shuffles() {
    long4 a =  l4_set_4x(-3, 18, -19, -2);
    int4 b = l4_cvt_i4(a);
    assert(i4_all_eq(b, i4_set_4x(-3, 18, -19, -2)));
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_f4_abs);
    PRINT_RUN(test_f4_signmask);
    PRINT_RUN(test_f4_or);
    PRINT_RUN(test_f4_xor);

    PRINT_RUN(test_from_vecs);
    PRINT_RUN(test_add);
    PRINT_RUN(test_dot);

    PRINT_RUN(test_f4_xor);
    PRINT_RUN(test_f4_or);
    PRINT_RUN(test_f4_scale);
    PRINT_RUN(test_broadcast);

    PRINT_RUN(test_d4_set_i8);
    PRINT_RUN(test_i4);
    PRINT_RUN(test_i4_blend);
    PRINT_RUN(test_shuffles);
    return 0;
}
