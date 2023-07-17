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

int
npy_type_size(npy_type tp) {
    if (tp == NPY_INT32 || tp == NPY_FLOAT32) {
        return 4;
    }
    return 8;
}

static const char *
type_names[3] = {"int32", "int64", "float64"};

const char *
npy_type_name(npy_type tp) {
    return type_names[tp];
}

size_t
npy_n_elements(npy_data *me) {
    size_t tot = 1;
    for (int i = 0; i < me->n_dims; i++) {
        tot *= me->dims[i];
    }
    return tot;
}

double
npy_value_at_as_double(npy_data *me, size_t i) {
    npy_type tp = me->type;
    if (tp == NPY_INT32) {
        return (double)((int32_t *)me->data)[i];
    } else if (tp == NPY_INT64) {
        return (double)((int64_t *)me->data)[i];
    } else if (tp == NPY_FLOAT32) {
        return (double)((float *)me->data)[i];
    } else if (tp == NPY_FLOAT64) {
        return ((double *)me->data)[i];
    }
    assert(false);
}

npy_data *
npy_load(const char *fname) {
    npy_data *me = (npy_data *)malloc(sizeof(npy_data));
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
    const char *formats[] = {"i4", "i8", "f4", "f8"};
    npy_type types[] = {
        NPY_INT32,
        NPY_INT64,
        NPY_FLOAT32,
        NPY_FLOAT64
    };
    for (int k = 0; k < 3; k++) {
        char rbuf[128];
        if (fscanf(f, "%s", rbuf) != 1) {
            goto parse_error;
        }
        if (strstr(rbuf, "'descr':")) {
            if (fscanf(f, "%s", rbuf) != 1) {
                goto parse_error;
            }
            bool found = false;
            for (int i = 0; i < ARRAY_SIZE(formats); i++) {
                if (strstr(rbuf, formats[i])) {
                    me->type = types[i];
                    found = true;
                }
            }
            if (!found) {
                goto parse_error;
            }
        } else if (strstr(rbuf, "'fortran_order':")) {
            if (fscanf(f, "%s", rbuf) != 1) {
                goto parse_error;
            }
            if (strstr(rbuf, "False")) {
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
    size_t n_bytes = n_els * npy_type_size(me->type);

    me->data = (char *)malloc(n_bytes);
    if (fread(me->data, 1, n_bytes, f) != n_bytes) {
        goto truncation_error;
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
npy_free(npy_data *me) {
    if (me->data) {
        free(me->data);
    }
    free(me);
}
