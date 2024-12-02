// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"

static void
check_dims(tensor *me, int n_dims, int dims[]) {
    tensor_check_equal_dims(me->n_dims, me->dims,
                            n_dims, dims);
}

void
test_layer_stack_apply_lenet() {
    tensor_layer *layers[] = {
        tensor_layer_init_conv2d(3, 6, 5, 1, 0),
        tensor_layer_init_relu(),
        tensor_layer_init_max_pool2d(2, 2, 2, 0),
        tensor_layer_init_conv2d(6, 16, 5, 1, 0),
        tensor_layer_init_relu(),
        tensor_layer_init_max_pool2d(2, 2, 2, 0),
        tensor_layer_init_flatten(0),
        tensor_layer_init_linear(400, 120),
        tensor_layer_init_relu(),
        tensor_layer_init_linear(120, 84),
        tensor_layer_init_relu(),
        tensor_layer_init_linear(84, 10)
    };
    tensor_layer_stack *stack = tensor_layer_stack_init(
        ARRAY_SIZE(layers), layers,
        3, (int[]){3, 32, 32}
    );

    tensor *x0 = tensor_init(3, (int[]){3, 32, 32});

    tensor *x1 = tensor_layer_stack_apply_new(stack, x0);
    assert(x1);

    assert(x1->n_dims == 1);
    assert(x1->dims[0] == 10);

    tensor_free(x0);
    tensor_free(x1);
    tensor_layer_stack_free(stack);
}

void
test_lenet_layer_stack_apply_relu() {
    tensor_layer *layers[] = {
        tensor_layer_init_relu()
    };
    tensor_layer_stack *stack = tensor_layer_stack_init(
        ARRAY_SIZE(layers), layers,
        1, (int[]){8}
    );

    tensor *x0 = tensor_init_from_data(
        (float *)(float[8]){
            -1, 8., 5., 2., 8., -7, 6., 2.
        }, 1, (int[]){8});

    tensor *x1 = tensor_layer_stack_apply_new(stack, x0);

    tensor *expected = tensor_init_from_data(
        (float *)(float[8]){
            0, 8., 5., 2., 8., 0, 6., 2.
        }, 1, (int[]){8});
    assert(x1->n_dims == 1);
    tensor_check_equal(x1, expected, LINALG_EPSILON);
    tensor_layer_stack_free(stack);
    tensor_free(x0);
    tensor_free(x1);
    tensor_free(expected);
}

void
test_lenet_layer_stack() {
    tensor_layer *layers[] = {
        tensor_layer_init_conv2d(3, 6, 5, 1, 0),
        tensor_layer_init_relu(),
        tensor_layer_init_max_pool2d(2, 2, 2, 0),
        tensor_layer_init_conv2d(6, 16, 5, 1, 0),
        tensor_layer_init_relu(),
        tensor_layer_init_max_pool2d(2, 2, 2, 0),
        tensor_layer_init_flatten(0),
        tensor_layer_init_linear(400, 120),
        tensor_layer_init_relu(),
        tensor_layer_init_linear(120, 84),
        tensor_layer_init_relu(),
        tensor_layer_init_linear(84, 10)
    };
    tensor_layer_stack *stack = tensor_layer_stack_init(
        ARRAY_SIZE(layers), layers,
        3, (int[]){3, 32, 32}
    );
    tensor_layer_stack_print(stack);
    tensor_layer_stack_free(stack);
}

void
test_layer_stack_apply_conv2d() {
    float conv1_weight[2][2][2][2] = {
        {
            {
                {9, 0},
                {4, 1}
            },
            {
                {2, 5},
                {1, 4}
            }
        },
        {
            {
                {0, 6},
                {0, 6}
            },
            {
                {6, 6},
                {0, 4}
            }
        }
    };
    float conv1_bias[2] = {4,  8};
    float expected_data[2][4][4] = {
        {
            {138.,  59., 117., 126.},
            {153., 179., 160., 135.},
            { 58.,  39., 120., 185.},
            {125., 191., 142., 131.}
        },
        {
            {92.,  44., 138., 182.},
            {130., 152., 182., 124.},
            { 86.,  56., 130., 176.},
            {148., 164., 164.,  80.}
        }
    };
    float x0_data[2][8][8] = {
        {
            {7., 5., 2., 0., 4., 2., 4., 7.},
            {7., 1., 3., 1., 2., 8., 2., 6.},
            {9., 7., 8., 1., 8., 5., 8., 4.},
            {2., 3., 5., 6., 5., 8., 0., 2.},
            {3., 0., 0., 3., 2., 5., 5., 2.},
            {1., 8., 1., 0., 9., 4., 9., 4.},
            {4., 3., 9., 6., 2., 2., 5., 0.},
            {2., 4., 9., 6., 9., 5., 8., 0.}
        },
        {
            {2., 0., 2., 1., 0., 9., 6., 4.},
            {2., 9., 3., 3., 0., 4., 4., 9.},
            {0., 9., 5., 6., 8., 8., 5., 5.},
            {4., 2., 1., 9., 0., 0., 2., 5.},
            {4., 1., 1., 0., 3., 3., 9., 9.},
            {2., 0., 5., 6., 5., 8., 9., 6.},
            {5., 6., 4., 4., 7., 8., 5., 5.},
            {1., 8., 0., 9., 1., 6., 3., 3.}
        }
    };
    tensor_layer *layers[] = {
        tensor_layer_init_conv2d_from_data(
            2, 2, 2, 2, 0,
            (float *)conv1_weight, (float *)conv1_bias
        ),
    };
    tensor_layer_stack *stack = tensor_layer_stack_init(
        ARRAY_SIZE(layers), layers,
        3, (int[]){2, 8, 8}
    );
    assert(tensor_n_elements(stack->src_buf) == 128);

    tensor *x0 = tensor_init_3d(2, 8, 8);
    tensor_copy_data(x0, x0_data);
    tensor *expected = tensor_init_3d(2, 4, 4);
    tensor_copy_data(expected, expected_data);
    tensor *x1 = tensor_layer_stack_apply_new(stack, x0);
    assert(x1);

    tensor_check_equal(x1, expected, LINALG_EPSILON);

    tensor_layer_stack_free(stack);
    tensor_free(x0);
    tensor_free(x1);
    tensor_free(expected);
}

void
test_lenet_layers() {
    tensor_layer *fc1 = tensor_layer_init_linear(400, 120);
    tensor_layer *fc2 = tensor_layer_init_linear(120, 84);
    tensor_layer *fc3 = tensor_layer_init_linear(84, 10);

    tensor_layer *conv1 = tensor_layer_init_conv2d(3, 6, 5, 1, 0);
    tensor_layer *conv2 = tensor_layer_init_conv2d(6, 16, 5, 1, 0);

    tensor_layer *max_pool2d = tensor_layer_init_max_pool2d(2, 2, 2, 0);
    tensor_layer *relu = tensor_layer_init_relu();
    tensor_layer *flatten = tensor_layer_init_flatten(0);


    // Run input through layers
    tensor *x0 = tensor_init(3, (int[]){3, 32, 32});

    tensor *x1 = tensor_layer_apply_new(conv1, x0);
    check_dims(x1, 3, (int[]){6, 28, 28});

    tensor *x2 = tensor_layer_apply_new(relu, x1);
    check_dims(x2, 3, (int[]){6, 28, 28});

    tensor *x3 = tensor_layer_apply_new(max_pool2d, x2);
    check_dims(x3, 3, (int[]){6, 14, 14});

    tensor *x4 = tensor_layer_apply_new(conv2, x3);
    check_dims(x4, 3, (int[]){16, 10, 10});

    tensor *x5 = tensor_layer_apply_new(max_pool2d, x4);
    check_dims(x5, 3, (int[]){16, 5, 5});

    tensor *x6 = tensor_layer_apply_new(flatten, x5);
    check_dims(x6, 1, (int[]){400});

    tensor *x7 = tensor_layer_apply_new(fc1, x6);
    check_dims(x7, 1, (int[]){120});

    tensor *x8 = tensor_layer_apply_new(fc2, x7);
    check_dims(x8, 1, (int[]){84});

    tensor *x9 = tensor_layer_apply_new(fc3, x8);
    check_dims(x9, 1, (int[]){10});

    tensor_layer_free(fc1);
    tensor_layer_free(fc2);
    tensor_layer_free(fc3);
    tensor_layer_free(conv1);
    tensor_layer_free(conv2);
    tensor_layer_free(max_pool2d);
    tensor_layer_free(relu);
    tensor_layer_free(flatten);

    tensor_free(x0);
    tensor_free(x1);
    tensor_free(x2);
    tensor_free(x3);
    tensor_free(x4);
    tensor_free(x5);
    tensor_free(x6);
    tensor_free(x7);
    tensor_free(x8);
    tensor_free(x9);
}

int
main(int argc, char *argv[]) {
    rand_init(1234);
    PRINT_RUN(test_lenet_layer_stack);
    PRINT_RUN(test_layer_stack_apply_conv2d);
    PRINT_RUN(test_lenet_layers);
    PRINT_RUN(test_lenet_layer_stack_apply_relu);
    PRINT_RUN(test_layer_stack_apply_lenet);
}
