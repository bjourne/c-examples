// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "datatypes/int-array.h"

void
test_pretty_print() {
    // Array with rowstride 1024
    int *arr = (int *)malloc(sizeof(int) * 1024 * 10);
    for (int i = 0; i < 1024 * 10; i++) {
        arr[i] = 0xffffff;
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            arr[i * 1024 + j] = i * j;
        }
    }
    int1d_pretty_print_table(arr, 10, 10, 1024, 0, NULL);
    free(arr);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_pretty_print);
    return 0;
}
