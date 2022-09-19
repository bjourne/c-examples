// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"

#define SIZE 4096

void
test_mul_perf() {
    int a_rows = SIZE;
    int a_cols = SIZE;
    int b_rows = SIZE;
    int b_cols = SIZE;

    tensor *a = tensor_init(2, (int[]){a_rows, a_cols});
    tensor *b = tensor_init(2, (int[]){b_rows, b_cols});
    tensor *c = tensor_init(2, (int[]){a_rows, b_cols});

    tensor_randrange(a, 100);
    tensor_randrange(b, 100);

    tensor_multiply(a, b, c);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_mul_perf);
}
