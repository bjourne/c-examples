// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"

void
test_crash() {
    int N = 64;
    int K = 256;
    int M = 256;
    tensor *a = tensor_init_2d(N, K);
    tensor *b = tensor_init_2d(K, M);
    tensor *c = tensor_init_2d(N, M);
    tensor_multiply(a, b, c);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void
test_crash2() {
    int N = 615;
    int K = 373;
    int M = 383;
    tensor *a = tensor_init_2d(N, K);
    tensor *b = tensor_init_2d(K, M);
    tensor *c = tensor_init_2d(N, M);
    tensor_multiply_w_params(a, b, c, 21);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void
test_mul_perf() {
    for (int n = 1; n < 8; n++) {
        int N = n * 32;
        for (int k = 1; k < 8; k++) {
            int K = k * 32;
            for (int m = 1; m < 8; m++) {
                int M = m * 32;

                tensor *a = tensor_init_2d(N, K);
                tensor *b = tensor_init_2d(K, M);
                tensor *c = tensor_init_2d(N, M);
                tensor *c_ref = tensor_init_2d(N, M);

                tensor_fill_rand_range(a, 5);
                tensor_fill_rand_range(b, 5);

                tensor_multiply(a, b, c);
                tensor_multiply_ref(a, b, c_ref);
                tensor_check_equal(c, c_ref, LINALG_EPSILON);

                tensor_free(a);
                tensor_free(b);
                tensor_free(c);
                tensor_free(c_ref);
            }
        }
    }
}

void
test_arbitrary_sizes() {
    int b0 = 200;
    int r = 500;
    int N = b0 + rand_n(r);
    int K = b0 + rand_n(r);
    int M = b0 + rand_n(r);

    printf("N = %d, K = %d, M = %d\n", N, K, M);
    tensor *a = tensor_init_2d(N, K);
    tensor *b = tensor_init_2d(K, M);
    tensor *c = tensor_init_2d(N, M);
    tensor *c_ref = tensor_init_2d(N, M);

    tensor_fill_rand_range(a, 10);
    tensor_fill_rand_range(b, 10);

    tensor_multiply(a, b, c);
    tensor_multiply_ref(a, b, c_ref);

    tensor_check_equal(c, c_ref, LINALG_EPSILON);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(c_ref);
}


void
test_multiply() {
    tensor *as[] = {
        tensor_init_from_data(
            (float *)(float[]){-1, 0, 3, 5},
            2, (int[]){2, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){15, 2, 3, 4, 5, 6, 7, 8, 9},
            2, (int[]){3, 3}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){4, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){2, 4}
        )
    };
    tensor *bs[] = {
        tensor_init_from_data(
            (float *)(float[]){-1, 0, 3, 5},
            2, (int[]){2, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){10, 11, 12, 13, 14, 15, 16, 17, 18},
            2, (int[]){3, 3}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){2, 4}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){4, 2}
        ),
    };
    tensor *cs[] = {
        tensor_init(2, (int[]){2, 2}),
        tensor_init(2, (int[]){3, 3}),
        tensor_init(2, (int[]){4, 4}),
        tensor_init(2, (int[]){2, 2})
    };
    tensor *c_exps[] = {
        tensor_init_from_data(
            (float *)(float[]){1, 0, 12, 25},
            2, (int[]){2, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){224, 244, 264, 201, 216, 231, 318, 342, 366},
            2, (int[]){3, 3}
        ),
        tensor_init_from_data(
            (float *)(float[]){
                4, 5, 6, 7,
                12, 17, 22, 27,
                20, 29, 38, 47,
                28, 41, 54, 67
            },
            2, (int[]){4, 4}
        ),
        tensor_init_from_data(
            (float *)(float[]){28, 34, 76, 98},
            2, (int[]){2, 2}
        )
    };
    for (size_t i = 0; i < ARRAY_SIZE(as); i++) {
        tensor *a = as[i];
        tensor *b = bs[i];
        tensor *c = cs[i];
        tensor *c_exp = c_exps[i];
        tensor_multiply(a, b, c);
        tensor_check_equal(c, c_exp, LINALG_EPSILON);
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(c_exp);
    }
}

void
test_multiply_big() {
    int dim = 512;
    tensor *a = tensor_init(2, (int[]){dim, dim});
    tensor *b = tensor_init(2, (int[]){dim, dim});
    tensor *c = tensor_init(2, (int[]){dim, dim});
    tensor *c_exp = tensor_init(2, (int[]){dim, dim});
    tensor_fill_const(c_exp, 0.0);

    tensor_fill_rand_range(a, 10.0);
    tensor_fill_rand_range(b, 10.0);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                float av = a->data[dim * i + k];
                float bv = b->data[dim * k + j];
                c_exp->data[dim * i + j] += av * bv;
            }
        }
    }
    tensor_multiply(a, b, c);
    tensor_check_equal(c, c_exp, LINALG_EPSILON);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(c_exp);
}



void
perf_test_multiply() {
    int SIZE = 2048;
    //int SIZE = 8192*2;
    tensor *a = tensor_init(2, (int[]){SIZE, SIZE});
    tensor *b = tensor_init(2, (int[]){SIZE, SIZE});
    tensor *c = tensor_init(2, (int[]){SIZE, SIZE});

    tensor_fill_range(a, 0.0f);
    tensor_fill_range(b, 0.0f);

    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    tensor_multiply(a, b, c);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double delta = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
        (end.tv_sec  - begin.tv_sec);
    double gflops = (long)SIZE * (long)SIZE * (long)SIZE
        / (delta * 1000.0 * 1000.0 * 1000.0);
    printf("%.6lfs, %.2lf gflops\n", delta, gflops);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}


int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_crash);
    PRINT_RUN(test_crash2);
    //PRINT_RUN(test_mul_perf);
    PRINT_RUN(test_arbitrary_sizes);

    PRINT_RUN(test_multiply);
    PRINT_RUN(test_multiply_big);
    perf_test_multiply();
}
