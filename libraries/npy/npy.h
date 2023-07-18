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

    // Type and size (in bytes) of elements
    char type;
    int el_size;

    // Dimensions
    int dims[NPY_MAX_N_DIMS];
    int n_dims;

    // Data
    char *data;

    // Error
    npy_error error_code;
} npy_arr;

size_t npy_n_elements(npy_arr *me);
npy_arr *npy_load(const char *fname);
void npy_free(npy_arr *me);
double npy_value_at_as_double(npy_arr *me, size_t i);

#endif
