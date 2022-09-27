// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"

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

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_mul_perf);
    PRINT_RUN(test_arbitrary_sizes);
}
