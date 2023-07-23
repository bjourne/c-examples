// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "npy/npy.h"

static void
print_dims(npy_arr *arr) {
    int *dims = arr->dims;
    int n_dims = arr->n_dims;
    printf("(");
    for (int i = 0; i < n_dims - 1; i++) {
        printf("%d, ", dims[i]);
    }
    printf("%d)", dims[n_dims - 1]);
}

int
main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Need name of .npy file.\n");
        return 1;
    }
    char *fname = argv[1];
    npy_arr *arr = npy_load(fname);
    npy_error err = arr->error_code;

    if (err != NPY_ERR_NONE) {
        printf("Error %d while reading file '%s'.\n", err, fname);
        return 1;
    }
    printf("Version   : %d.%d\n", arr->ver_maj, arr->ver_min);
    printf("Type      : %c%d\n", arr->type, arr->el_size);
    printf("Dimensions: ");
    print_dims(arr);
    printf("\n");

    int n_columns = 100;
    struct winsize w;
    if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &w)) {
        n_columns = w.ws_col;
    }
    npy_pp *pp = npy_pp_init(1, n_columns, ", ");

    npy_pp_print_arr(pp, arr);

    npy_pp_free(pp);
    npy_free(arr);
    return 0;
}
