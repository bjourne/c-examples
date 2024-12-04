// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>


#define TENSOR_MAX_N_DIMS   10

// Address alignment for data buffers. Since tensors are supposed to
// be pretty big, wasting a few bytes shouldn't matter much.
#define TENSOR_ADDRESS_ALIGNMENT    64

typedef enum {
    TENSOR_ERR_NONE = 0,
    TENSOR_ERR_FILE_NOT_FOUND,
    TENSOR_ERR_NOT_A_PNG_FILE,
    TENSOR_ERR_UNSUPPORTED_PNG_TYPE,
    TENSOR_ERR_PNG_ERROR,
    TENSOR_ERR_WRONG_DIMENSIONALITY,
    TENSOR_ERR_TOO_BIG
} tensor_error_type;

typedef enum {
    TENSOR_LAYER_LINEAR = 0,
    TENSOR_LAYER_CONV2D,
    TENSOR_LAYER_MAX_POOL2D,
    TENSOR_LAYER_RELU,
    TENSOR_LAYER_FLATTEN
} tensor_layer_type;

typedef enum {
    TENSOR_UNARY_OP_SOFTMAX = 0,
    TENSOR_UNARY_OP_EXP,
    TENSOR_UNARY_OP_ADD,
    TENSOR_UNARY_OP_DIV,
    TENSOR_UNARY_OP_MAX
} tensor_unary_op;

typedef enum {
    TENSOR_BINARY_OP_ADD = 0,
    TENSOR_BINARY_OP_MUL
} tensor_binary_op;

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
    int input_dims[TENSOR_MAX_N_DIMS];

    int *layers_n_dims;
    int **layers_dims;

    // Two buffers are needed to propagate the tensors through the
    // stack since some layers can't process in place.
    tensor *src_buf;
    tensor *dst_buf;

} tensor_layer_stack;

// Init & free
tensor *tensor_init(int n_dims, int dims[]);
tensor *tensor_init_1d(int x);
tensor *tensor_init_2d(int x, int y);
tensor *tensor_init_3d(int x, int y, int z);
tensor *tensor_init_4d(int x, int y, int z, int w);

// Utility
long tensor_n_elements(tensor *me);
void tensor_flatten(tensor *me, int from);
void tensor_set_dims(tensor *me, int n_dims, int dims[]);
int tensor_padded_strided_dim(int s_dim, int f_dim, int pad, int stride);

// Shouldn't use this one
tensor *tensor_init_from_data(float *data, int n_dims, int dims[]);
tensor *tensor_init_copy(tensor *orig);

void tensor_free(tensor *t);

// Copy data
void tensor_copy_data(tensor *me, void *addr);

// Checking
bool tensor_check_equal(tensor *t1, tensor *t2, float eps);
void tensor_check_equal_contents(tensor *t1, tensor *t2, float eps);
void tensor_check_equal_dims(int n_dims1, int dims1[],
                             int n_dims2, int dims2[]);
void tensor_check_dims(tensor *t, int n_dims, ...);

// Printing
void tensor_print(tensor *me, bool print_header,
                  int n_decimals, int n_columns,
                  char *sep);


// Unary ops
void tensor_unary(tensor *src, tensor *dst,
                  tensor_unary_op op, float scalar);

// Scans
void tensor_scan(tensor *src, tensor *dst, tensor_binary_op op,
                 bool exclusive, float seed);

// Fills
void tensor_fill_const(tensor *t, float v);
void tensor_fill_rand_range(tensor *t, float high);
void tensor_fill_range(tensor *me, float start);

// Conv2d
void tensor_conv2d(tensor *weight, tensor *bias,
                   int stride, int padding,
                   tensor *src, tensor *dst);

tensor *tensor_conv2d_new(tensor *weight, tensor *bias,
                          int stride, int padding,
                          tensor *src);
void
tensor_im2col(tensor *src, tensor *dst,
              int stride_y, int stride_x,
              int pad_y, int pad_x);

tensor *
tensor_im2col_new(tensor *src,
                  int fy_dim, int fx_dim,
                  int stride_y, int stride_x,
                  int pad_y, int pad_x);

// MaxPool2d
void tensor_max_pool2d(int kernel_height, int kernel_width,
                       int stride, int padding,
                       tensor *src, tensor *dst);
tensor *tensor_max_pool2d_new(int kernel_height, int kernel_width,
                              int stride, int padding,
                              tensor *src);

// Linear
void tensor_linear(tensor *weights, tensor *bias,
                   tensor *src, tensor *dst);
tensor *tensor_linear_new(tensor *weights, tensor *bias, tensor *src);

// Transpose
void tensor_transpose(tensor *src, tensor *dst);
tensor *tensor_transpose_new(tensor *src);

#ifdef HAVE_PNG
// Png support
tensor *tensor_read_png(char *filename);
bool tensor_write_png(tensor *me, char *filename);
#endif

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
