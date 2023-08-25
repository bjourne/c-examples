// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "npy/npy.h"

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

    char buf[256];
    npy_format_dims(arr, buf);
    printf("Dimensions: %s\n", buf);

    int n_columns = 100;
    struct winsize w;
    if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &w)) {
        n_columns = w.ws_col;
    }
    npy_pp *pp = npy_pp_init(1, n_columns, " ");

    npy_pp_print_arr(pp, arr);

    npy_pp_free(pp);
    npy_free(arr);
    return 0;
}
