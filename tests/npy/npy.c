// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "npy/npy.h"

void
test_load_missing() {
    npy_arr *arr = npy_load("tests/npy/eeeh");
    assert(arr->error_code == NPY_ERR_FILE_NOT_FOUND);
    npy_free(arr);
}

void
test_load_uint8() {
    npy_arr *arr = npy_load("tests/npy/uint8.npy");
    assert(arr->n_dims == 1);
    assert(arr->dims[0] == 100);
    npy_free(arr);
}

void
test_pretty_print() {
    npy_arr *arr = npy_load("tests/npy/uint8.npy");
    npy_free(arr);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_load_missing);
    PRINT_RUN(test_load_uint8);
    PRINT_RUN(test_pretty_print);
}
