// Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_LAYERS_H
#define TENSORS_LAYERS_H

#include "tensors.h"

typedef enum {
    TENSOR_LAYER_LINEAR = 0,
    TENSOR_LAYER_CONV2D,
    TENSOR_LAYER_MAX_POOL2D,
    TENSOR_LAYER_RELU,
    TENSOR_LAYER_FLATTEN
} tensor_layer_type;

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
    int stride;
    int padding;
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

typedef struct {
    int n_layers;
    tensor_layer **layers;

    // Need to record input and output dimensions somewhere.
    int input_n_dims;
    int *input_dims;
    //int input_dims[TENSOR_MAX_N_DIMS];

    int *layers_n_dims;
    int **layers_dims;

    // Two buffers are needed to propagate the tensors through the
    // stack since some layers can't process in place.
    tensor *src_buf;
    tensor *dst_buf;

} tensor_layer_stack;

// Layer abstraction
tensor_layer *tensor_layer_init_linear(int in, int out);
tensor_layer *tensor_layer_init_relu();
tensor_layer *tensor_layer_init_flatten(int from);
tensor_layer *tensor_layer_init_max_pool2d(int kernel_height, int kernel_width,
                                           int stride, int padding);
tensor_layer *tensor_layer_init_conv2d(int in_chans, int out_chans,
                                       int kernel_size,
                                       int stride,
                                       int padding);
tensor_layer *tensor_layer_init_conv2d_from_data(int in_chans, int out_chans,
                                                 int kernel_size,
                                                 int stride, int padding,
                                                 float *weight_data, float *bias_data);


void tensor_layer_free(tensor_layer *me);

tensor *tensor_layer_apply_new(tensor_layer *me, tensor *input);

// Stack abstraction

// The stack takes ownership of the layers.
tensor_layer_stack *
tensor_layer_stack_init(int n_layers, tensor_layer *layers[],
                        int input_n_dims,
                        int *input_dims);
tensor *tensor_layer_stack_apply_new(tensor_layer_stack *me, tensor *input);
void tensor_layer_stack_free(tensor_layer_stack *me);

void tensor_layer_stack_print(tensor_layer_stack *me);


#endif
