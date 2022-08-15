// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

#define TENSOR_MAX_N_DIMS   10

typedef enum {
    TENSOR_ERR_NONE = 0,
    TENSOR_ERR_FILE_NOT_FOUND,
    TENSOR_ERR_NOT_A_PNG_FILE,
    TENSOR_ERR_PNG_ERROR,
    TENSOR_ERR_WRONG_DIMENSIONALITY
} tensor_err_t;

typedef struct {
    int dims[TENSOR_MAX_N_DIMS];
    int n_dims;
    float *data;
    tensor_err_t error_code;
} tensor;

tensor *tensor_read_png(char *filename);

bool tensor_write_png(tensor *me, char *filename);

void tensor_free(tensor *t);



#endif
