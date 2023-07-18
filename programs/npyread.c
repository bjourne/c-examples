// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "npy/npy.h"

static void
pp_indent(int n) {
    for (int i = 0; i < n; i++) {
        printf(" ");
    }
}

typedef struct {
    npy_arr *arr;
    char *fmt;
    bool break_lines;
    size_t value_idx;
    int items_per_line;
} npy_pp;

void
npy_pp_print_dims(npy_pp *me) {
    int *dims = me->arr->dims;
    int n_dims = me->arr->n_dims;
    printf("(");
    for (int i = 0; i < n_dims - 1; i++) {
        printf("%d, ", dims[i]);
    }
    printf("%d)", dims[n_dims - 1]);
}

static void
pp_value(npy_pp *me, size_t row_idx) {
    npy_arr *arr = me->arr;
    char tp = arr->type;
    char *data = arr->data;
    int n_dims = arr->n_dims;
    int v = arr->el_size;
    char *fmt = me->fmt;
    size_t i = me->value_idx;

    if (me->break_lines &&
        row_idx % me->items_per_line == 0 &&
        row_idx > 0) {
        printf("\n");
        pp_indent(n_dims);
    }
    if (tp == 'i') {
        if (v == 1) {
            printf(fmt, ((int8_t *)data)[i]);
        } else if (v == 4) {
            printf(fmt, ((int32_t *)data)[i]);
        } else if (v == 8) {
            printf(fmt, ((int64_t *)data)[i]);
        } else {
            assert(false);
        }
    } else if (tp == 'f') {
        if (v == 4) {
            printf(fmt, ((float *)data)[i]);
        } else if (v == 8) {
            printf(fmt, ((double *)data)[i]);
        } else {
            assert(false);
        }
    } else {
        const char *s = data + v * i;
        for (int k = 0; k < v; k++) {
            unsigned char c = *s;
            if (c == 0) {
                break;
            }
            putchar(c);
            s++;
        }
    }
    me->value_idx++;
}

static void
pp_row(npy_pp *me, bool is_first) {
    npy_arr *arr = me->arr;
    int n_dims = arr->n_dims;
    if (!is_first) {
        pp_indent(n_dims - 1);
    }
    printf("[");
    size_t cnt = me->arr->dims[n_dims - 1];
    for (int i = 0; i < cnt - 1; i++) {
        pp_value(me, i);
        printf(", ");
    }
    pp_value(me, cnt - 1);
    printf("]");
}

static void
pp_rec(npy_pp *me, int dim_idx, bool is_first) {
    npy_arr *arr = me->arr;
    if (dim_idx == arr->n_dims - 1) {
        pp_row(me, is_first);
    } else {
        if (!is_first) {
            pp_indent(dim_idx);
        }
        printf("[");
        int n_els = arr->dims[dim_idx];
        for (int i = 0; i < n_els - 1; i++) {
            pp_rec(me, dim_idx + 1, i == 0);
            printf(",\n");
        }
        pp_rec(me, dim_idx + 1, n_els == 1);
        printf("]");
    }
    if (dim_idx == 0) {
        printf("\n");
    }
}

void
npy_pp_print_arr(npy_pp *me) {
    me->value_idx = 0;
    pp_rec(me, 0, true);
}

static int
npy_pp_set_fmt(npy_arr *arr, char *fmt, int n_decimals) {
    char tp = arr->type;
    if (tp == 'S') {
        sprintf(fmt, "%%%ds", arr->el_size);
        return arr->el_size;
    }
    double max = npy_value_at_as_double(arr, 0);
    double min = 0;
    for (int i = 1; i < npy_n_elements(arr); i++) {
        double at = npy_value_at_as_double(arr, i);
        if (at < min) {
            min = at;
        } else if (at > max) {
            max = at;
        }
    }
    int max_width = (int)ceil(log10(max));
    int min_width = (int)ceil(log10(-min)) + 1;
    int width = MAX(max_width, min_width);
    if (arr->type == 'i') {
        sprintf(fmt, "%%%dd", width);
    } else  {
        width += 1 + n_decimals;
        sprintf(fmt, "%%%d.%df", width, n_decimals);
    }
    return width;
}

npy_pp *
npy_pp_init(npy_arr *arr, int n_decimals) {
    npy_pp *me = (npy_pp *)malloc(sizeof(npy_pp));
    me->fmt = malloc(256);
    me->arr = arr;

    int width = npy_pp_set_fmt(arr, me->fmt, n_decimals);

    // Estimate line-width
    int n_els = arr->dims[arr->n_dims - 1];
    int line_length = (n_els - 1) * (width + 2) + width + 2;
    me->break_lines = line_length > 150;
    me->items_per_line = 150 / (width + 2);
    return me;
}

void
npy_pp_free(npy_pp *me) {
    free(me->fmt);
    free(me);
}

int
main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Need name of .npy file.\n");
        return 1;
    }
    char *fname = argv[1];
    npy_arr *npy = npy_load(fname);
    npy_error err = npy->error_code;

    if (err != NPY_ERR_NONE) {
        printf("Error %d while reading file '%s'.\n", err, fname);
        return 1;
    }
    printf("Version   : %d.%d\n", npy->ver_maj, npy->ver_min);
    printf("Type      : %c%d\n", npy->type, npy->el_size);
    printf("Dimensions: ");
    npy_pp *pp = npy_pp_init(npy, 2);
    npy_pp_print_dims(pp);
    printf("\n");
    npy_pp_print_arr(pp);

    npy_pp_free(pp);
    npy_free(npy);
    return 0;
}
