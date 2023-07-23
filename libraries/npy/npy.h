// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// For reading binary .npy files.
#ifndef NPY_H
#define NPY_H

#include <stdbool.h>

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

// Pretty printing
typedef struct {
    // Printing style
    int n_decimals;
    int n_columns;
    const char *sep;

    // Computed
    char fmt[256];
    int n_items_per_line;
    bool break_lines;

    // Set during printing
    size_t value_idx;
    npy_arr *arr;
} npy_pp;

npy_pp *npy_pp_init(int n_decimals, int line_width, const char *sep);
void npy_pp_free(npy_pp *me);
void npy_pp_print_arr(npy_pp *me, npy_arr *arr);

#endif
