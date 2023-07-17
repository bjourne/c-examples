// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// For reading binary .npy files.
#ifndef NPY_H
#define NPY_H

#define NPY_MAX_N_DIMS   10

typedef enum {
    NPY_ERR_NONE = 0,
    NPY_ERR_FILE_NOT_FOUND,
    NPY_ERR_TRUNCATED,
    NPY_ERR_BAD_MAGIC,
    NPY_ERR_HEADER_PARSE
} npy_error;

typedef enum {
    NPY_INT32,
    NPY_INT64,
    NPY_FLOAT32,
    NPY_FLOAT64
} npy_type;

typedef struct {
    // Version
    int ver_maj, ver_min;

    // Value type
    npy_type type;

    // Dimensions
    int dims[NPY_MAX_N_DIMS];
    int n_dims;

    // Data
    char *data;

    // Error
    npy_error error_code;
} npy_data;

size_t npy_n_elements(npy_data *me);
npy_data *npy_load(const char *fname);
void npy_free(npy_data *me);

int npy_type_size(npy_type tp);
const char *npy_type_name(npy_type tp);

double npy_value_at_as_double(npy_data *me, size_t i);

#endif
