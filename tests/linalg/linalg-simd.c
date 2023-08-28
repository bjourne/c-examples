// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"
#include "linalg/linalg-simd.h"
#include "random/random.h"

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

void
test_mix_d4_i4_and_blend() {
    int4 cnt = i4_load((int32_t[]){4, 0, 0, 0});
    double4 x =  d4_load((double[]){30.0, 18.0, 7.0, 2.0});
    double4 thr = d4_set_1x(15.0);
    int4 cnt_rst = i4_set_1x(5);

    int4 cnt_next = i4_sub(cnt, i4_1());
    double4 event_d4 = d4_cmp_gte(x, thr);
    int4 event_i4 = d4_cvt_i4(event_d4);

    x = d4_tern(i4_cvt_d4(cnt_next),
                d4_tern(event_d4, d4_0(), x),
                d4_0());

    cnt_next = i4_tern(cnt_next,
                  i4_tern(event_i4, cnt_rst, i4_0()),
                  cnt_next);

    printf("cnt next: ");
    i4_print(cnt_next);
    printf("\n");
    printf("x: ");
    d4_print(x, 1);
    printf("\n");
}

#define N_G (1024 * 10)
#define N_S (8 * N_G)
#define N_T (5 * 10 * 1000)

typedef struct {
    double x, y, z;
    int8_t c;
} item;

static uint32_t
run_interleaved_loop(
    item *ptr,
    double c1, double c2, double c3,
    double thr, uint32_t rst) {
    uint32_t n_evs = 0;
    for (uint32_t i = 0; i < N_S; i++) {
        if (!ptr->c) {
            ptr->x = ptr->x * c1 + ptr->y * c2 + ptr->z * c3;
            if (ptr->x >= thr) {
                n_evs++;
                ptr->x = 0.0;
                ptr->c = rst;
            }
        } else {
            ptr->c -= 1;
        }
        ptr++;
    }
    return n_evs;
}

static uint32_t
run_scalar_loop(
    double *x_ptr, double *y_ptr, int8_t *c_ptr,
    double *o_ptr,
    double c1, double c2, double c3,
    double thr, uint32_t rst) {
    uint32_t n_evs = 0;
    for (uint32_t i = 0; i < 8; i++) {
        double z = o_ptr[i];
        for (uint32_t j = 0; j < N_G; j++) {
            double x = *x_ptr;
            double y = *y_ptr;
            int64_t c = *c_ptr;
            if (!c) {
                x = x * c1 + y * c2 + z * c3;
                if (x >= thr) {
                    n_evs++;
                    *x_ptr = 0.0;
                    *c_ptr = rst;
                } else {
                    *x_ptr = x;
                }
            } else {
                *c_ptr = c - 1;
            }
            x_ptr++;
            y_ptr++;
            c_ptr++;
        }
    }
    return n_evs;
}

static uint32_t
run_avx2_loop(
    double *x_ptr, double *y_ptr, int64_t *c_ptr, double *o_ptr,
    double c1, double c2, double c3,
    double thr, uint32_t rst) {
    double4 r_c1 = d4_set_1x(c1);
    double4 r_c2 = d4_set_1x(c2);
    double4 r_c3 = d4_set_1x(c3);
    long4 r_rst = l4_set_1x(rst);
    double4 r_thr = d4_set_1x(thr);
    uint32_t n_evs = 0;
    for (uint32_t i = 0; i < 8; i++) {
        double4 z = d4_set_1x(o_ptr[i]);
        for (uint32_t j = 0; j < N_G; j += 4) {
            double4 x = d4_load(x_ptr);
            double4 y = d4_load(y_ptr);
            long4 c = l4_load(c_ptr);
            c = l4_sub(c, l4_1());

            double4 t1 = d4_mul(x, r_c1);
            double4 t2 = d4_mul(y, r_c2);
            double4 t3 = d4_mul(z, r_c3);
            x = d4_add(d4_add(t1, t2), t3);

            double4 ev = d4_cmp_gte(x, r_thr);

            x = d4_tern((double4)c, d4_andnot(ev, x), d4_0());
            c = l4_tern(c, l4_and((long4)ev, r_rst), c);

            n_evs += d4_count_mask(ev);
            d4_store(x, x_ptr);
            l4_store(c, c_ptr);

            x_ptr += 4;
            y_ptr += 4;
            c_ptr += 4;
        }
    }
    return n_evs;
}

static uint32_t
run_sse_loop(
    double *x_ptr, double *y_ptr, int64_t *c_ptr, double *o_ptr,
    double c1, double c2, double c3,
    double thr, uint32_t rst) {
    double2 r_c1 = d2_set_1x(c1);
    double2 r_c2 = d2_set_1x(c2);
    double2 r_c3 = d2_set_1x(c3);
    long2 r_rst = l2_set_1x(rst);
    double2 r_thr = d2_set_1x(thr);
    uint32_t n_evs = 0;
    for (uint32_t i = 0; i < 8; i++) {
        double2 z = d2_set_1x(o_ptr[i]);
        for (uint32_t j = 0; j < N_G; j += 2) {
            double2 x = d2_load(x_ptr);
            double2 y = d2_load(y_ptr);
            long2 c = l2_load(c_ptr);
            c = l2_sub(c, l2_1());

            double2 t1 = d2_mul(x, r_c1);
            double2 t2 = d2_mul(y, r_c2);
            double2 t3 = d2_mul(z, r_c3);
            x = d2_add(d2_add(t1, t2), t3);

            double2 ev = d2_cmp_gte(x, r_thr);
            uint32_t ev_mask = d2_movemask(ev);

            x = d2_tern((double2)c, d2_andnot(ev, x), d2_0());
            c = l2_tern(c, l2_and((long2)ev, r_rst), c);
            n_evs += __builtin_popcount(ev_mask);
            d2_store(x, x_ptr);
            l2_store(c, c_ptr);

            x_ptr += 2;
            y_ptr += 2;
            c_ptr += 2;
        }
    }
    return n_evs;
}

static void
run_test(uint32_t tp, uint32_t *n_evs, uint64_t *nanos) {
    double *xs = malloc_aligned(0x40, N_S * sizeof(double));
    double *ys = malloc_aligned(0x40, N_S * sizeof(double));
    double *zs = malloc_aligned(0x40, N_S * sizeof(double));
    int64_t *cs8 = malloc_aligned(0x40, N_S * sizeof(int64_t));
    int8_t *cs1 = malloc_aligned(0x40, N_S * sizeof(int8_t));
    item *items = malloc_aligned(0x40, sizeof(item) * N_S);
    double *os = malloc_aligned(0x40, sizeof(double) * 8);

    rnd_pcg32_seed(1001, 370);
    for (uint32_t i = 0; i < 8; i++) {
        os[i] = (double)rnd_pcg32_rand_range(1000) / 5;
    }
    for (uint32_t i = 0; i < 8; i++) {
        for (uint32_t j = 0; j < N_G; j++) {
            uint32_t k = i * N_G + j;
            xs[k] = (double)rnd_pcg32_rand_range(1000) / 100;
            ys[k] = (double)rnd_pcg32_rand_range(1000) / 5;
            zs[k] = os[i];
            cs8[k] = 0;
            cs1[k] = 0;
            items[k].x = xs[k];
            items[k].y = ys[k];
            items[k].z = zs[k];
            items[k].c = cs8[k];
        }
    }
    double c1 = 0.990049834;
    double c2 = 0.000398007;
    double c3 = 0.000360672;
    uint32_t rst = 20;
    double thr = 10.0;

    *n_evs = 0;
    uint64_t start = nano_count();
    if (tp == 0) {
        for (uint32_t t = 0; t < N_T; t++) {
            *n_evs += run_sse_loop(xs, ys, cs8, os,
                                   c1, c2, c3,
                                   thr, rst);
        }
    } else if (tp == 1) {
        for (uint32_t t = 0; t < N_T; t++) {
            *n_evs += run_avx2_loop(xs, ys, cs8, os,
                                    c1, c2, c3,
                                    thr, rst);
        }
    } else if (tp == 2) {
        for (uint32_t t = 0; t < N_T; t++) {
            *n_evs += run_scalar_loop(xs, ys, cs1, os,
                                      c1, c2, c3,
                                      thr, rst);
        }
    } else if (tp  == 3) {
        for (uint32_t t = 0; t < N_T; t++) {
            *n_evs += run_interleaved_loop(items,
                                           c1, c2, c3,
                                           thr, rst);
        }
    }
    *nanos = nano_count() - start;
    free(items);
    free(xs);
    free(ys);
    free(zs);
    free(os);
    free(cs8);
    free(cs1);
}

void
benchmark_tern_etc() {
    //uint64_t nanos;
    assert(N_S % 4 == 0);

    // On my laptop the difference is quite small between the scalar
    // loop and the avx2-coded one.
    uint32_t cnts[4];
    char *names[4] = {"sse", "avx2", "scalar", "interleaved"};
    for (size_t i = 0; i < ARRAY_SIZE(cnts); i++) {
        uint64_t nanos;
        run_test(i, &cnts[i], &nanos);
        printf("%-15s %9d %6.2f\n",
               names[i], cnts[i], (double)nanos / (1000 * 1000 * 1000));
    }
    assert(cnts[0] == cnts[1] && cnts[1] == cnts[2]);
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
    PRINT_RUN(test_mix_d4_i4_and_blend);

    PRINT_RUN(benchmark_tern_etc);

    return 0;
}
