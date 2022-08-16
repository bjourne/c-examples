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
} tensor_error_type;

typedef  enum {
    TENSOR_LAYER_LINEAR = 0,
    TENSOR_LAYER_CONV2D,
    TENSOR_LAYER_MAX_POOL2D,
    TENSOR_LAYER_RELU,
    TENSOR_LAYER_FLATTEN
} tensor_layer_type;

typedef struct {
    int dims[TENSOR_MAX_N_DIMS];
    int n_dims;
    float *data;
    tensor_error_type error_code;
} tensor;

typedef struct {
    int from;
} tensor_layer_flatten;

typedef struct {
    tensor *weight;
    tensor *bias;
} tensor_layer_linear;

typedef struct {
    int kernel_width;
    int kernel_height;
} tensor_layer_max_pool2d;

typedef struct {
    tensor *weight;
    tensor *bias;
    int stride;
    int padding;
} tensor_layer_conv2d;

typedef struct {
    tensor_layer_type type;
    union {
        tensor_layer_linear linear;
        tensor_layer_conv2d conv2d;
        tensor_layer_max_pool2d max_pool2d;
        tensor_layer_flatten flatten;
    };
} tensor_layer;

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
                          tensor *src);

// MaxPool2d
void tensor_max_pool2d(tensor *src,
                       int kernel_height, int kernel_width,
                       tensor *dst,
                       int stride, int padding);
tensor *tensor_max_pool2d_new(tensor *src,
                              int kernel_height, int kernel_width,
                              int stride, int padding);

// Linear
void tensor_linear(tensor *weights, tensor *bias,
                   tensor *src, tensor *dst);
tensor *tensor_linear_new(tensor *weights, tensor *bias, tensor *src);

// Scalar ops
void tensor_relu(tensor *src);
void tensor_fill(tensor *t, float v);
bool tensor_check_equal(tensor *t1, tensor *t2);
void tensor_randrange(tensor *t1, int high);

// Png support
tensor *tensor_read_png(char *filename);
bool tensor_write_png(tensor *me, char *filename);

// Layer abstraction
tensor_layer *tensor_layer_init_linear(int in, int out);
tensor_layer *tensor_layer_init_relu();
tensor_layer *tensor_layer_init_flatten(int from);
tensor_layer *tensor_layer_init_max_pool_2d(int kernel_height,
                                            int kernel_width);
tensor_layer *tensor_layer_init_conv2d(int in_chans, int out_chans,
                                       int kernel_size,
                                       int stride,
                                       int padding);
void tensor_layer_free(tensor_layer *me);




#endif
