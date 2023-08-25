// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "datatypes/common.h"
#include "npy/npy.h"

void
test_load_missing() {
    npy_arr *arr = npy_load("tests/npy/eeeh");
    assert(arr->error_code == NPY_ERR_OPEN_FILE);
    npy_free(arr);
}

void
test_load_uint8() {
    npy_arr *arr = npy_load("tests/npy/uint8.npy");
    assert(arr->n_dims == 1);
    assert(arr->dims[0] == 100);
    assert(arr->type == 'u');
    npy_free(arr);
}

void
test_pretty_print() {
    npy_arr *arr = npy_load("tests/npy/uint8.npy");
    npy_arr *arr2 = npy_load("tests/npy/rands.npy");
    npy_arr *arr3 = npy_load("tests/npy/empty.npy");

    int n_columns = 100;
    struct winsize w;
    if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &w)) {
        n_columns = w.ws_col;
    }
    npy_pp *pp = npy_pp_init(1, n_columns, " ");
    npy_pp_print_arr(pp, arr);
    pp->n_columns = 78;
    pp->sep = ", ";
    npy_pp_print_arr(pp, arr2);
    npy_pp_print_arr(pp, arr3);

    npy_pp_free(pp);
    npy_free(arr);
    npy_free(arr2);
    npy_free(arr3);
}

void
test_pretty_print_bytes() {
    npy_arr *arr = npy_load("tests/npy/bytes.npy");
    npy_pp *pp = npy_pp_init(1, 100, " ");
    npy_pp_print_arr(pp, arr);
    npy_pp_free(pp);
    npy_free(arr);
}

void
test_load_and_save() {
    npy_arr *orig = npy_load("tests/npy/uint8.npy");
    assert(orig->n_dims == 1);
    assert(orig->dims[0] == 100);
    assert(npy_save(orig, "tmp.npy") == NPY_ERR_NONE);

    npy_arr *copy = npy_load("tmp.npy");
    assert(copy->n_dims == 1);
    assert(copy->dims[0] == 100);

    npy_free(copy);
    npy_free(orig);
}


int
main(int argc, char *argv[]) {
    PRINT_RUN(test_load_missing);
    PRINT_RUN(test_load_uint8);
    PRINT_RUN(test_pretty_print);
    PRINT_RUN(test_pretty_print_bytes);
    PRINT_RUN(test_load_and_save);
}
