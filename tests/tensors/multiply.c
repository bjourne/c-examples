// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"

#define SIZE 1024

void
test_mul_perf() {
    int a_rows = SIZE;
    int a_cols = SIZE;
    int b_rows = SIZE;
    int b_cols = SIZE;

    tensor *a = tensor_init(2, (int[]){a_rows, a_cols});
    tensor *b = tensor_init(2, (int[]){b_rows, b_cols});
    tensor *c = tensor_init(2, (int[]){a_rows, b_cols});

    tensor_fill_rand_ints(a, 100);
    tensor_fill_rand_ints(b, 100);

    tensor_multiply(a, b, c);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void
test_arbitrary_sizes() {
    rand_init(0);

    int a_rows = 1000 + rand_n(1000);
    int a_cols = 1000 + rand_n(1000);
    int b_rows = a_cols;
    int b_cols = 1000 + rand_n(1000);

    tensor *a = tensor_init(2, (int[]){a_rows, a_cols});
    tensor *b = tensor_init(2, (int[]){b_rows, b_cols});
    tensor *c = tensor_init(2, (int[]){a_rows, b_cols});
    tensor *c_ref = tensor_init(2, (int[]){a_rows, b_cols});

    tensor_fill_rand_ints(a, 10);
    tensor_fill_rand_ints(b, 10);

    tensor_multiply(a, b, c);
    tensor_multiply_ref(a, b, c_ref);

    tensor_check_equal(c, c_ref, LINALG_EPSILON);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(c_ref);
}

void
test_linearize_tiles() {
    tensor *src = tensor_init(2, (int[]){4, 4});
    tensor *dst = tensor_init(2, (int[]){4, 4});
    tensor *dst_ref = tensor_init_from_data(
        (float *)(float[]){
            1, 2, 5, 6,
            3, 4, 7, 8,
            9, 10, 13, 14,
            11, 12, 15, 16},
        2, (int[]){4, 4});

    tensor_fill_range(src, 1.0);
    tensor_linearize_tiles(src, dst, 2, 2);
    tensor_check_equal(dst, dst_ref, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(dst_ref);
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
    for (int i = 0; i < ARRAY_SIZE(as); i++) {
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
    int dim = 1024;
    tensor *a = tensor_init(2, (int[]){dim, dim});
    tensor *b = tensor_init(2, (int[]){dim, dim});
    tensor *c = tensor_init(2, (int[]){dim, dim});
    tensor *c_exp = tensor_init(2, (int[]){dim, dim});
    tensor_fill_const(c_exp, 0.0);

    tensor_fill_rand_ints(a, 10.0);
    tensor_fill_rand_ints(b, 10.0);

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


int
main(int argc, char *argv[]) {
    PRINT_RUN(test_mul_perf);
    PRINT_RUN(test_arbitrary_sizes);
    PRINT_RUN(test_linearize_tiles);
    //PRINT_RUN(test_multiply);
    //PRINT_RUN(test_multiply_big);
}
