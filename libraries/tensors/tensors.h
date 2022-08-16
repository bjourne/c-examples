// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

#define TENSOR_MAX_N_DIMS   10

typedef enum {
    TENSOR_ERR_NONE = 0,
    TENSOR_ERR_FILE_NOT_FOUND,
    TENSOR_ERR_NOT_A_PNG_FILE,
    TENSOR_ERR_UNSUPPORTED_PNG_TYPE,
    TENSOR_ERR_PNG_ERROR,
    TENSOR_ERR_WRONG_DIMENSIONALITY
} tensor_err_t;

typedef struct {
    int dims[TENSOR_MAX_N_DIMS];
    int n_dims;
    float *data;
    tensor_err_t error_code;
} tensor;

tensor *tensor_init(int n_dims, ...);
tensor *tensor_init_from_data(float *data, int n_dims, ...);
tensor *tensor_init_filled(float v, int n_dims, ...);
void tensor_free(tensor *t);

// Utility
int tensor_n_elements(tensor *me);
void tensor_flatten(tensor *me, int from);

// Conv2d
void tensor_conv2d(tensor *weight, tensor *bias,
                   int stride, int padding,
                   tensor *src, tensor *dst);

tensor *tensor_conv2d_new(tensor *weight, tensor *bias,
                          int stride, int padding,
                          tensor  *src);

// MaxPool2d
void tensor_max_pool2d(tensor *src,
                       int kernel_height, int kernel_width,
                       tensor *dst,
                       int stride, int padding);
tensor *tensor_max_pool2d_new(tensor *src,
                              int kernel_height, int kernel_width,
                              int stride, int padding);

// Linear
void tensor_linear(tensor *src, tensor *weights, tensor *bias,
                   tensor *dst);
tensor *tensor_linear_new(tensor *src, tensor *weights, tensor *bias);

// Scalar ops
void tensor_relu(tensor *src);
void tensor_fill(tensor *t, float v);
bool tensor_check_equal(tensor *t1, tensor *t2);
void tensor_randrange(tensor *t1, int high);

// Png support
tensor *tensor_read_png(char *filename);
bool tensor_write_png(tensor *me, char *filename);




#endif
