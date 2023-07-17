// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "npy/npy.h"

static void
indent(int n) {
    for (int i = 0; i < n; i++) {
        printf(" ");
    }
}

static char *
print_value(npy_data *me, char *ptr, char *fmt) {
    npy_type tp = me->type;
    if (tp == NPY_INT32) {
        printf(fmt, *(int32_t *)ptr);
    } else if (tp == NPY_INT64) {
        printf(fmt, *(int64_t *)ptr);
    } else if (tp == NPY_FLOAT32) {
        printf(fmt, *(float *)ptr);
    } else {
        printf(fmt, *(double *)ptr);
    }
    return ptr + npy_type_size(tp);
}

static char *
inner_row(npy_data *me, char *ptr, int idx, bool is_first, char *fmt) {
    if (!is_first) {
        indent(idx);
    }
    printf("[");
    for (int i = 0; i < me->dims[idx] - 1; i++) {
        if (i > 0 && i % 20 == 0) {
            printf("\n");
            indent(idx + 1);
        }
        ptr = print_value(me, ptr, fmt);
        printf(", ");
    }
    ptr = print_value(me, ptr, fmt);
    printf("]");
    return ptr;
}

static char *
pretty_print(npy_data *me, int idx, char *ptr, bool is_first,
             char *fmt) {
    if (idx == me->n_dims - 1) {
        ptr = inner_row(me, ptr, idx, is_first, fmt);
    } else {
        if (!is_first) {
            indent(idx);
        }
        printf("[");
        int n_els = me->dims[idx];
        for (int i = 0; i < n_els - 1; i++) {
            ptr = pretty_print(me, idx + 1, ptr, i == 0, fmt);
            printf(",\n");
        }
        ptr = pretty_print(me, idx + 1, ptr, n_els == 1, fmt);
        printf("]");
    }
    if (idx == 0) {
        printf("\n");
    }
    return ptr;
}

static void
max_and_min(npy_data *me, double *max, double *min) {
    *max = 1;
    *min = -1;
    for (int i = 0; i < npy_n_elements(me); i++) {
        double at = npy_value_at_as_double(me, i);
        if (at < *min) {
            *min = at;
        } else if (at > *max) {
            *max = at;
        }
    }
}

void
npy_pretty_print(npy_data *me) {
    double max, min;
    max_and_min(me, &max, &min);

    int max_digits = (int)ceil(log10(max));
    int min_digits = (int)ceil(log10(-min)) + 1;
    int width = max_digits > min_digits ? max_digits : min_digits;

    char fmt[256];
    if (me->type == NPY_INT32) {
        sprintf(fmt, "%%%dd", width);
    } else if (me->type == NPY_INT64) {
        sprintf(fmt, "%%%dld", width);
    } else {
        sprintf(fmt, "%%%d.%df", width + 4, 3);
    }
    pretty_print(me, 0, me->data, false, fmt);
}

int
main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Need name of .npy file.\n");
        return 1;
    }
    char *fname = argv[1];
    npy_data *npy = npy_load(fname);
    npy_error err = npy->error_code;

    if (err != NPY_ERR_NONE) {
        printf("Error %d while reading file '%s'.\n", err, fname);
        return 1;
    }
    printf("Version   : %d.%d\n", npy->ver_maj, npy->ver_min);
    printf("Type      : %s\n", npy_type_name(npy->type));
    printf("Dimensions: (");
    for (int i = 0; i < npy->n_dims - 1; i++) {
        printf("%d, ", npy->dims[i]);
    }
    printf("%d)\n", npy->dims[npy->n_dims - 1]);
    npy_pretty_print(npy);
    npy_free(npy);
    return 0;
}
