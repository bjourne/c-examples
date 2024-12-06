// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_LAYERS_H
#define TENSORS_LAYERS_H

#include "tensors.h"

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
