// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "npy/npy.h"

static bool
consume(FILE *f, char *str) {
    for (int i = 0; i < strlen(str); i++) {
        char c = fgetc(f);
        if (feof(f) || c != str[i]) {
            return false;
        }
    }
    return true;
}
static bool
consume_until(FILE *f, char c) {
    while (!feof(f)) {
        if (fgetc(f) == c) {
            return true;
        }
    }
    return false;
}

size_t
npy_n_elements(npy_arr *me) {
    size_t tot = 1;
    for (int i = 0; i < me->n_dims; i++) {
        tot *= me->dims[i];
    }
    return tot;
}

double
npy_value_at_as_double(npy_arr *me, size_t i) {
    char tp = me->type;
    int v = me->el_size;
    if (tp == 'i') {
        if (v == 1) {
            return (double)((int8_t *)me->data)[i];
        }
        else if (v == 4) {
            return (double)((int32_t *)me->data)[i];
        } else if (v == 8) {
            return (double)((int64_t *)me->data)[i];
        }
        assert(false);
    } else if (tp == 'f') {
        if (v == 4) {
            return (double)((float *)me->data)[i];
        } else if (v == 8) {
            return ((double *)me->data)[i];
        }
        assert(false);
    }
    assert(false);
}

static void
transpose_data(npy_arr *me) {
    assert(me->n_dims == 2);
    size_t n_els = npy_n_elements(me);
    int *dims = me->dims;
    int rows = dims[0];
    int cols = dims[1];
    int el_size = me->el_size;
    char *new = malloc(el_size * n_els);
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            char *from = me->data + el_size * (x * rows + y);
            char *to = new + el_size * (y * cols + x);
            memcpy(to, from, el_size);
            from += el_size;
        }
    }
    free(me->data);
    me->data = new;
}

npy_arr *
npy_load(const char *fname) {
    npy_arr *me = (npy_arr *)malloc(sizeof(npy_arr));
    me->error_code = NPY_ERR_NONE;
    me->n_dims = 0;
    me->data = NULL;

    FILE *f = fopen(fname, "rb");
    if (!f) {
        me->error_code = NPY_ERR_FILE_NOT_FOUND;
        return me;
    }
    char buf[10];
    if (fread(buf, 1, 10, f) != 10) {
        goto truncation_error;
    }
    if (memcmp(buf, "\x93NUMPY", 6)) {
        me->error_code = NPY_ERR_BAD_MAGIC;
        goto done;
    }
    me->ver_maj = buf[6];
    me->ver_min = buf[7];

    if (!consume(f, "{")) {
        goto truncation_error;
    }
    bool fortran_order = false;
    for (int k = 0; k < 3; k++) {
        char rbuf[128];
        if (fscanf(f, "%s", rbuf) != 1) {
            goto parse_error;
        }
        if (strstr(rbuf, "'descr':")) {
            if (fscanf(f, "%s", rbuf) != 1) {
                goto parse_error;
            }
            me->type = rbuf[2];
            if (sscanf(&rbuf[3], "%d", &me->el_size) != 1) {
                goto parse_error;
            }
        } else if (strstr(rbuf, "'fortran_order':")) {
            if (fscanf(f, "%s", rbuf) != 1) {
                goto parse_error;
            }
            if (strstr(rbuf, "False")) {
                fortran_order = false;
            } else if (strstr(rbuf, "True")) {
                fortran_order = true;
            } else {
                goto parse_error;
            }
        } else if (strstr(rbuf, "'shape':")) {
            if (!consume(f, " (")) {
                goto parse_error;
            }
            int dim;
            while (fscanf(f, " %d,", &dim) == 1) {
                me->dims[me->n_dims] = dim;
                me->n_dims++;
            }
            if (!consume(f, "),")) {
                goto parse_error;
            }
        } else {
            goto parse_error;
        }
    }
    if (!consume_until(f, '\n')) {
        goto parse_error;
    }
    size_t n_els = npy_n_elements(me);
    size_t n_bytes = n_els * me->el_size;
    me->data = (char *)malloc(n_bytes);
    if (fread(me->data, 1, n_bytes, f) != n_bytes) {
        goto truncation_error;
    }
    if (fortran_order) {
        transpose_data(me);
    }
 done:
    fclose(f);
    return me;
 parse_error:
    me->error_code = NPY_ERR_HEADER_PARSE;
    fclose(f);
    return me;
 truncation_error:
    me->error_code = NPY_ERR_TRUNCATED;
    fclose(f);
    return me;
}

void
npy_free(npy_arr *me) {
    if (me->data) {
        free(me->data);
    }
    free(me);
}
