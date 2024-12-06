// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "conv2d.h"
#include "layers.h"

////////////////////////////////////////////////////////////////////////
// Layer abstraction
////////////////////////////////////////////////////////////////////////
tensor_layer *
tensor_layer_init_relu() {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_RELU;
    return me;
}

tensor_layer *
tensor_layer_init_flatten(int from) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_FLATTEN;
    me->flatten.from = from;
    return me;
}

tensor_layer *
tensor_layer_init_linear(int in, int out) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_LINEAR;

    tensor *weight = tensor_init_2d(out, in);
    tensor *bias = tensor_init_1d(out);

    me->linear.weight = weight;
    me->linear.bias = bias;

    return me;
}

tensor_layer *
tensor_layer_init_max_pool2d(int kernel_height, int kernel_width,
                             int stride, int padding) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_MAX_POOL2D;
    me->max_pool2d.kernel_width = kernel_width;
    me->max_pool2d.kernel_height = kernel_height;
    me->max_pool2d.stride = stride;
    me->max_pool2d.padding = padding;
    return me;
}

tensor_layer *
tensor_layer_init_conv2d(int in_chans, int out_chans,
                         int kernel_size,
                         int stride, int padding) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_CONV2D;

    tensor *weight = tensor_init_4d(out_chans, in_chans,
                                    kernel_size, kernel_size);
    tensor *bias = tensor_init_1d(out_chans);
    me->conv2d.weight = weight;
    me->conv2d.bias = bias;
    me->conv2d.stride = stride;
    me->conv2d.padding = padding;
    return me;
}

tensor_layer *
tensor_layer_init_conv2d_from_data(int in_chans, int out_chans,
                                   int kernel_size,
                                   int stride, int padding,
                                   float *weight_data, float *bias_data) {
    tensor_layer *l = tensor_layer_init_conv2d(in_chans, out_chans,
                                               kernel_size,
                                               stride, padding);
    tensor *weight = l->conv2d.weight;
    tensor *bias = l->conv2d.bias;

    tensor_copy_data(weight, weight_data);
    tensor_copy_data(bias, bias_data);
    return l;
}

int
tensor_layer_n_params(tensor_layer *me) {
    tensor_layer_type t = me->type;
    if (t == TENSOR_LAYER_CONV2D || t == TENSOR_LAYER_LINEAR) {
        tensor *w, *b;
        if (t == TENSOR_LAYER_CONV2D) {
            w = me->conv2d.weight;
            b = me->conv2d.bias;
        } else {
            w = me->linear.weight;
            b = me->linear.bias;
        }
        return tensor_n_elements(w) + tensor_n_elements(b);
    }
    return 0;
}

void
tensor_layer_free(tensor_layer *me) {
    if (me->type == TENSOR_LAYER_LINEAR)  {
        tensor_free(me->linear.weight);
        tensor_free(me->linear.bias);
    }  else if (me->type == TENSOR_LAYER_CONV2D) {
        tensor_free(me->conv2d.weight);
        tensor_free(me->conv2d.bias);
    }
    free(me);
}

tensor *
tensor_layer_apply_new(tensor_layer *me, tensor *input) {
    tensor_layer_type t = me->type;
    if (t == TENSOR_LAYER_LINEAR) {
        return tensor_linear_new(me->linear.weight, me->linear.bias, input);
    } else if (t == TENSOR_LAYER_CONV2D) {
        return tensor_conv2d_new(me->conv2d.weight, me->conv2d.bias,
                                 me->conv2d.stride, me->conv2d.padding,
                                 input);
    } else if (t == TENSOR_LAYER_RELU) {
        tensor *output = tensor_init_copy(input);
        tensor_unary(output, output, TENSOR_UNARY_OP_MAX, 0);
        return output;
    } else if (t == TENSOR_LAYER_MAX_POOL2D) {
        return tensor_max_pool2d_new(me->max_pool2d.kernel_width,
                                     me->max_pool2d.kernel_height,
                                     me->max_pool2d.stride,
                                     me->max_pool2d.padding, input);
    } else if (t == TENSOR_LAYER_FLATTEN) {
        tensor *output = tensor_init_copy(input);
        tensor_flatten(output, me->flatten.from);
        return output;
    } else {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////
// Stack abstraction
////////////////////////////////////////////////////////////////////////
tensor_layer_stack *
tensor_layer_stack_init(int n_layers, tensor_layer **layers,
                        int input_n_dims, int *input_dims) {

    // Easiest way to compute dimensions of intermediate layers is to
    // run a dummy tensor through the stack.
    tensor *input = tensor_init(input_n_dims, input_dims);

    tensor_layer_stack *me = malloc(sizeof(tensor_layer_stack));
    me->n_layers = n_layers;
    me->layers = layers;
    tensor_dims_clone(input_n_dims, input_dims, &me->input_n_dims, &me->input_dims);

    // Layers' dims
    me->layers_n_dims = (int *)malloc(sizeof(int) * n_layers);
    me->layers_dims = (int **)malloc(sizeof(int *) * n_layers);

    long max = tensor_dims_count(input_n_dims, input_dims);
    for (int i = 0; i < n_layers; i++) {
        tensor *output = tensor_layer_apply_new(me->layers[i], input);
        int n_dims = output->n_dims;
        int *dims = output->dims;

        long this = tensor_dims_count(n_dims, dims);
        max = MAX(this, max);

        tensor_dims_clone(n_dims, dims, &me->layers_n_dims[i], &me->layers_dims[i]);
        tensor_free(input);
        input = output;
    }
    tensor_free(input);
    me->src_buf = tensor_init_1d(max);
    me->dst_buf = tensor_init_1d(max);
    return me;
}

void
tensor_layer_stack_free(tensor_layer_stack *me) {
    for (int i = 0; i < me->n_layers; i++) {
        tensor_layer_free(me->layers[i]);
        free(me->layers_dims[i]);
    }
    tensor_free(me->src_buf);
    tensor_free(me->dst_buf);
    free(me->input_dims);
    free(me->layers_dims);
    free(me->layers_n_dims);
    free(me);
}

// The point is to avoid redundant mallocs and copies.
tensor *
tensor_layer_stack_apply_new(tensor_layer_stack *me, tensor *inp) {
    tensor_dims_check_equal(
        inp->n_dims, inp->dims,
        me->input_n_dims, me->input_dims
    );
    tensor *src = me->src_buf;
    tensor *dst = me->dst_buf;

    tensor_dims_copy(inp->n_dims, inp->dims, &src->n_dims, src->dims);
    tensor_copy_data(src, inp->data);

    for (int i = 0; i < me->n_layers; i++) {
        tensor_layer *l = me->layers[i];
        tensor_layer_type t = l->type;

        tensor_dims_copy(me->layers_n_dims[i], me->layers_dims[i],
                         &dst->n_dims, dst->dims);

        // Content is in dst after
        bool swap = false;
        if (t == TENSOR_LAYER_RELU) {
            tensor_unary(src, src, TENSOR_UNARY_OP_MAX, 0);
        } else if (t == TENSOR_LAYER_CONV2D) {
            tensor_conv2d(l->conv2d.weight, l->conv2d.bias,
                          l->conv2d.stride, l->conv2d.padding,
                          src, dst);
            swap = true;
        } else if (t == TENSOR_LAYER_MAX_POOL2D) {
            tensor_max_pool2d(l->max_pool2d.kernel_width,
                              l->max_pool2d.kernel_height,
                              l->max_pool2d.stride,
                              l->max_pool2d.padding,
                              src, dst);
            swap = true;
        } else if (t == TENSOR_LAYER_FLATTEN) {
            tensor_flatten(src, l->flatten.from);
        } else if (t == TENSOR_LAYER_LINEAR) {
            tensor_linear(l->linear.weight, l->linear.bias, src, dst);
            swap = true;
        } else {
            assert(false);
        }
        if (swap) {
            tensor *tmp = src;
            src = dst;
            dst = tmp;
        }
    }
    return tensor_init_copy(src);
}

static const char *
layer_name(tensor_layer_type t) {
    if (t == TENSOR_LAYER_LINEAR) {
        return "Linear";
    } else if (t == TENSOR_LAYER_RELU) {
        return "ReLU";
    } else if (t == TENSOR_LAYER_MAX_POOL2D) {
        return "MaxPool2D";
    } else if (t == TENSOR_LAYER_CONV2D) {
        return "Conv2D";
    } else if (t == TENSOR_LAYER_FLATTEN)  {
        return "Flatten";
    }
    return "Unknown";
}

static void
layer_details(char *buf, tensor_layer *l) {
    tensor_layer_type t = l->type;
    if (t == TENSOR_LAYER_CONV2D || t == TENSOR_LAYER_LINEAR) {
        tensor *w, *b;
        if (t == TENSOR_LAYER_CONV2D) {
            w = l->conv2d.weight;
            b = l->conv2d.bias;
        } else {
            w = l->linear.weight;
            b = l->linear.bias;
        }
        tensor_dims_to_string(w->n_dims, w->dims, buf);
        strcat(buf, ", ");
        tensor_dims_to_string(b->n_dims, b->dims, buf);
    }
}

void
tensor_layer_stack_print(tensor_layer_stack *me) {
    int n_layers = me->n_layers;
    int n_params = 0;
    for (int i = 0; i < n_layers; i++) {
        n_params += tensor_layer_n_params(me->layers[i]);
    }
    printf("Input: ");
    tensor_dims_print(me->input_n_dims, me->input_dims);
    printf(", Layers: %d, Params: %d\n", n_layers, n_params);
    for (int i = 0; i < n_layers; i++) {
        char buf[256] =  {0};
        tensor_layer *l = me->layers[i];
        tensor_layer_type t = l->type;
        layer_details(buf, l);
        printf("  %-10s %-20s: ", layer_name(t), buf);
        tensor_dims_print(me->layers_n_dims[i], me->layers_dims[i]);
        printf("\n");
    }
}
