// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"

char *fname = NULL;

static float
mat_5x5_1[5][5] = {
    {0, 1, 4, 3, 2},
    {1, 3, 4, 0, 4},
    {2, 2, 4, 1, 1},
    {2, 1, 3, 3, 2},
    {0, 0, 3, 1, 0}
};

// Transposed 5x5_1
static float
mat_5x5_2[5][5] = {
    {0, 1, 2, 2, 0},
    {1, 3, 2, 1, 0},
    {4, 4, 4, 3, 3},
    {3, 0, 1, 3, 1},
    {2, 4, 1, 2, 0}
};

static float
mat_2x4[2][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8}
};

static float
mat_2x4_t[4][2] = {
    {1, 5},
    {2, 6},
    {3, 7},
    {4, 8}
};

static float
arr_5[5] = {
    -4.0, 0.0, -20.0, 3.0, 2.0
};
static float
arr_5_after_relu[5] = {
    0.0, 0.0, 0.0, 3.0, 2.0
};
static int
dim_5[1] = {5};
static int
dim_10[1] = {10};

static float
arr_10_triangular_numbers[11] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55
};

void
test_from_png() {
    #ifdef HAVE_PNG
    tensor *t = tensor_read_png(fname);
    assert(t);
    assert(t->error_code == TENSOR_ERR_NONE);

    assert(tensor_write_png(t, "out_01.png"));
    assert(t->error_code == TENSOR_ERR_NONE);

    tensor_free(t);
    #endif
}

void
test_pick_channel() {
    #ifdef HAVE_PNG
    tensor *t1 = tensor_read_png(fname);
    assert(t1);
    assert(t1->error_code == TENSOR_ERR_NONE);

    int height = t1->dims[1];
    int width = t1->dims[2];

    tensor *t2 = tensor_init(3, (int[]){3, height, width});
    tensor_fill_const(t2, 0.0);
    for (int c = 0; c < 1; c++) {
        float *dst = &t2->data[c * height * width];
        memcpy(dst, t1->data, height * width * sizeof(float));
    }
    assert(tensor_write_png(t2, "out_02.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);

    tensor_free(t1);
    tensor_free(t2);
    #endif
}

void
test_conv2d() {
    #ifdef HAVE_PNG
    tensor *t1 = tensor_read_png(fname);
    assert(t1);
    assert(t1->error_code == TENSOR_ERR_NONE);

    float w = 1/8.0;

    // Blur red and green channels and filter blue channel.
    float weight_data[3][3][3][3] = {
        {
            {
                {w, w, w},
                {w, 0, w},
                {w, w, w}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            }
        },
        {
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {w, w, w},
                {w, 0, w},
                {w, w, w}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            }
        },
        {
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            }
        }
    };
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){3, 3, 3, 3});
    tensor *bias = tensor_init(1, (int[]){3});
    tensor_fill_const(bias, 0);

    tensor *t2 = tensor_conv2d_new(weight, bias, 1, 1, t1);

    assert(tensor_write_png(t2, "out_04.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);
    tensor_free(t2);
    tensor_free(t1);
    tensor_free(weight);
    #endif
}

void
test_conv2d_padded() {
    float weight_data[1][1][2][2] = {
        {
            {
                {4, 0},
                {4, 3}
            }
        }
    };
    float expected_data[1][4][4] = {
        {
            {13, 28, 32, 24},
            {18, 32, 35,  7},
            {19, 21, 37, 22},
            { 8, 13, 27, 16}
        }
    };
    tensor *src = tensor_init_from_data((float *)mat_5x5_1,
                                        3, (int[]){1, 5, 5});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 1, 2, 2});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){1, 4, 4});
    tensor *bias = tensor_init(1, (int[]){1});
    tensor_fill_const(bias, 0);

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_padded_2() {
    float src_data[1][5][5] = {
        {
            {2, 3, 1, 0, 1},
            {2, 0, 3, 0, 4},
            {2, 0, 1, 3, 1},
            {4, 1, 2, 3, 1},
            {3, 4, 1, 4, 4}
        }
    };
    float weight_data[1][1][2][2] = {
        {
            {
                {0, 2},
                {1, 2}
            }
        }
    };
    float expected_data[1][6][6] = {
        {
            { 4,  8,  5,  1,  2,  1},
            { 8,  8,  8,  3, 10,  4},
            { 8,  2,  8,  7, 13,  1},
            {12,  6,  7, 14,  7,  1},
            {14, 13, 10, 15, 14,  4},
            { 6,  8,  2,  8,  8,  0}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 5, 5});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 1, 2, 2});
    tensor *bias = tensor_init(1, (int[]){1});
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){1, 6, 6});

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 1, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_2channels() {
    float src_data[2][3][6] = {
        {
            {4, 2, 2, 2, 4, 4},
            {4, 0, 3, 0, 0, 3},
            {2, 3, 1, 0, 2, 4}
        },
        {
            {4, 2, 1, 1, 2, 3},
            {4, 4, 2, 3, 1, 3},
            {2, 1, 2, 1, 4, 1}
        }
    };
    float weight_data[1][2][1][1] = {
        {
            {
                {2}
            },
            {
                {3}
            }
        }
    };
    float expected_data[1][3][6] = {
        {
            {20, 10,  7,  7, 14, 17},
            {20, 12, 12,  9,  3, 15},
            {10,  9,  8,  3, 16, 11},
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){2, 3, 6});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 2, 1, 1});

    tensor *bias = tensor_init(1, (int[]){1});
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){1, 3, 6});

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_2x2channels() {
    float src_data[2][2][4] = {
        {
            {0, 2, 3, 3},
            {1, 2, 3, 1}
        },
        {
            {0, 2, 3, 1},
            {0, 0, 4, 0}
        }
    };
    float weight_data[2][2][1][1] = {
        {
            {
                {2}
            },
            {
                {3}
            }
        },
        {
            {
                {3}
            },
            {
                {4}
            }
        }
    };
    float expected_data[2][2][4] = {
        {
            { 0, 10, 15,  9},
            { 2,  4, 18,  2}
        },
        {
            { 0, 14, 21, 13},
            { 3,  6, 25,  3}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){2, 2, 4});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){2, 2, 1, 1});
    tensor *bias = tensor_init(1, (int[]){2});
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 2, 4});

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_strided() {
    float src_data[1][5][5] = {
        {
            {2, 4, 3, 3, 2},
            {4, 1, 3, 3, 4},
            {3, 1, 2, 2, 1},
            {3, 0, 2, 0, 0},
            {0, 2, 1, 0, 1}
        }
    };
    float weight_data[1][1][3][3] = {
        {
            {
                {2, 4, 1},
                {0, 0, 3},
                {2, 4, 4}
            }
        }
    };
    float expected_data[1][3][3] = {
        {
            {32, 35, 22},
            {32, 31, 22},
            {18,  8,  0}
        }
    };

    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 1, 3, 3});
    tensor *bias = tensor_init(1, (int[]){1});
    tensor_fill_const(bias, 0);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){1, 3, 3});
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 5, 5});
    tensor *dst = tensor_conv2d_new(weight, bias, 2, 1, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}


void
test_conv2d_3() {
    float src_data[1][10][3] = {
        {
            {4, 0, 4},
            {4, 2, 0},
            {1, 4, 2},
            {4, 1, 2},
            {1, 0, 2},
            {4, 4, 0},
            {2, 3, 0},
            {2, 4, 0},
            {0, 2, 1},
            {1, 0, 3}
        }
    };
    float weight_data[1][1][3][3] = {
        {
            {
                {1, 1, 1},
                {1, 0, 1},
                {1, 1, 1}
            }
        }
    };
    float expected_data[1][10][3] = {
        {
            { 6, 14,  2},
            {11, 19, 12},
            {15, 16,  9},
            { 7, 16,  9},
            {13, 18,  7},
            {10, 12,  9},
            {17, 16, 11},
            {11, 10, 10},
            { 9, 11,  9},
            { 2,  7,  3}
        }
    };

    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 10, 3});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 1, 3, 3});
    tensor *bias = tensor_init(1, (int[]){1});
    tensor_fill_const(bias, 0);
    tensor *dst = tensor_init(3, (int[]){1, 10, 3});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){1, 10, 3});

    tensor_conv2d(weight, bias, 1, 1, src, dst);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_uneven() {
    float src_data[1][2][4] = {
        {
            {4, 1, 4, 0},
            {0, 0, 0, 2}
        }
    };
    float weight_data[2][1][2][2] = {
        {
            {
                {1, 1},
                {4, 4}
            }
        },
        {
            {
                {3, 3},
                {1, 2}
            }
        }
    };
    float expected_data[2][1][3] = {
        {
            {5, 5, 12}
        },
        {
            {15, 15, 16}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 2, 4});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){2, 1, 2, 2});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 1, 3});
    tensor *bias = tensor_init(1, (int[]){2});
    tensor_fill_const(bias, 0);
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_uneven_strided() {
    float src_data[1][2][4] = {
        {
            {3, 0, 3, 1},
            {3, 4, 2, 2}
        }
    };
    float weight_data[2][1][2][2] = {
        {
            {
                {4, 1},
                {3, 3}
            }
        },
        {
            {
                {3, 3},
                {2, 0}
            }
        }
    };
    float expected_data[2][1][2] = {
        {
            {33, 25}
        },
        {
            {15, 16}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 2, 4});
    tensor *dst = tensor_init(3, (int[]){2, 1, 2});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){2, 1, 2, 2});
    tensor *bias = tensor_init(1, (int[]){2});
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 1, 2});

    tensor_conv2d(weight, bias, 2, 0, src, dst);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_conv2d_with_bias() {
    float src_data[2][5][5] = {
        {
            {5., 1., 1., 1., 8.},
            {6., 4., 6., 2., 5.},
            {0., 8., 7., 7., 4.},
            {8., 1., 2., 2., 0.},
            {0., 5., 4., 3., 0.},
        },
        {
            {9., 5., 8., 3., 0.},
            {0., 7., 7., 0., 5.},
            {8., 3., 7., 9., 2.},
            {5., 1., 1., 7., 8.},
            {5., 9., 9., 6., 0.}
        }
    };
    float weight_data[2][2][3][3] = {
        {
            {
                {3., 6., 2.},
                {9., 0., 6.},
                {7., 2., 2.}
            },
            {
                {7., 0., 2.},
                {6., 5., 1.},
                {6., 5., 9.}
            }
        },
        {
            {
                {7., 1., 4.},
                {1., 7., 4.},
                {3., 2., 5.}
            },
            {
                {2., 9., 6.},
                {0., 7., 8.},
                {8., 4., 4.}
            }
        }
    };
    float bias_data[2] = {7., 9.};
    float expected_data[2][3][3] = {
        {
            {397., 402., 395.},
            {293., 373., 413.},
            {433., 367., 316.}
        },
        {
            {478., 385., 327.},
            {429., 391., 346.},
            {310., 479., 431.}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){2, 5, 5});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){2, 2, 3, 3});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 3, 3});
    tensor *bias = tensor_init_from_data((float *)bias_data,
                                         1, (int[]){2});
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);


    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(expected);
}

void
test_max_pool_1() {
    float src_data[2][5][5] = {
        {
            {7, 5, 6, 1, 3},
            {9, 3, 3, 2, 8},
            {1, 1, 6, 1, 8},
            {1, 8, 2, 3, 3},
            {1, 0, 7, 2, 4}
        },
        {
            {5, 8, 0, 0, 1},
            {9, 5, 5, 2, 6},
            {9, 3, 7, 4, 3},
            {5, 5, 4, 3, 0},
            {8, 2, 1, 8, 8}
        }
    };
    float expected1_data[2][4][4] = {
        {
            {9., 6., 6., 8.},
            {9., 6., 6., 8.},
            {8., 8., 6., 8.},
            {8., 8., 7., 4.}
        },
        {
            {9., 8., 5., 6.},
            {9., 7., 7., 6.},
            {9., 7., 7., 4.},
            {8., 5., 8., 8.}
        }
    };
    float expected2_data[2][1][5] = {
        {
            {9., 8., 7., 3., 8.}
        },
        {
            {9., 8., 7., 8., 8.}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){2, 5, 5});
    tensor *expected1 = tensor_init_from_data((float *)expected1_data,
                                              3, (int[]){2, 4, 4});
    tensor *expected2 = tensor_init_from_data((float *)expected2_data,
                                              3, (int[]){2, 1, 5});
    tensor *dst1 = tensor_init(3, (int[]){2, 4, 4});
    tensor *dst2 = tensor_init(3, (int[]){2, 1, 5});

    tensor_max_pool2d(2, 2, 1, 0, src, dst1);
    tensor_check_equal(expected1, dst1, LINALG_EPSILON);

    tensor_max_pool2d(5, 1, 1, 0, src, dst2);
    tensor_check_equal(expected2, dst2, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst1);
    tensor_free(dst2);
    tensor_free(expected1);
    tensor_free(expected2);
}

void
test_max_pool_strided() {
    float src_data[2][5][5] = {
        {
            {6, 3, 1, 8, 2},
            {4, 9, 9, 7, 8},
            {5, 3, 7, 0, 0},
            {0, 0, 6, 0, 1},
            {3, 2, 9, 6, 6}
        },
        {
            {6, 7, 7, 9, 9},
            {0, 9, 3, 3, 8},
            {4, 1, 8, 1, 8},
            {9, 9, 2, 8, 8},
            {3, 7, 7, 9, 3}
        }
    };
    float expected_data[2][2][2] = {
        {
            {9, 9},
            {5, 7}
        },
        {
            {9, 9},
            {9, 8}
        }
    };
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){2, 5, 5});

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 2, 2});
    tensor *dst = tensor_max_pool2d_new(2, 2, 2, 0, src);
    tensor_check_equal(expected, dst, LINALG_EPSILON);
    tensor_free(src);
    tensor_free(expected);
}

void
test_max_pool_image() {
    #ifdef HAVE_PNG
    tensor *src = tensor_read_png(fname);
    assert(src);
    assert(src->error_code == TENSOR_ERR_NONE);
    for (int s = 1; s < 5; s++) {
        tensor *dst = tensor_max_pool2d_new(s, s, 1, 0, src);

        char fname_fmt[128];
        sprintf(fname_fmt, "out-max_pool-%02d.png", s - 1);
        assert(tensor_write_png(dst, fname_fmt));
        assert(dst->error_code == TENSOR_ERR_NONE);
        tensor_free(dst);
    }
    tensor_free(src);
    #endif
}

void
test_relu() {
    tensor *src = tensor_init_from_data(
        (float *)arr_5, 1, dim_5);
    tensor *expected = tensor_init_from_data(
        (float *)arr_5_after_relu, 1, dim_5);

    tensor_unary(src, src, TENSOR_UNARY_OP_RELU);
    tensor_check_equal(src, expected, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(expected);
}

void
test_linear() {
    tensor *src = tensor_init_from_data(
        (float *)(float[]){
            0, 2, 4, 5, 0, 1, 4, 4
        }, 1, (int[]){8});
    tensor *weight = tensor_init_from_data(
        (float *)(float[4][8]){
            {5, 6, 7, 6, 9, 3, 1, 9},
            {4, 9, 5, 0, 9, 9, 8, 7},
            {6, 8, 7, 4, 1, 9, 3, 1},
            {4, 1, 4, 8, 2, 0, 1, 1}
        }, 2, (int[]){4, 8});
    tensor *bias = tensor_init_from_data(
        (float *)(float[]){0, 0, 0, 0},
        1, (int[]){4});

    tensor *dst = tensor_init(1, (int[]){4});
    tensor *expected = tensor_init_from_data(
        (float *)(float[]){113, 107, 89, 66},
        1, (int[]){4});

    tensor_linear(weight, bias, src, dst);
    tensor_check_equal(dst, expected, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_linear_with_bias() {
    tensor *src = tensor_init_from_data(
        (float *)(float[]){
            0., 7., 6., 9., 6., 5., 0., 2., 7., 4.,
            2., 0., 3., 9., 7., 1., 4., 1., 0., 9.,
            0, 2, 4, 5, 0, 1, 4, 4
        }, 1, (int[]){20});
    tensor *weight = tensor_init_from_data(
        (float *)(float[10][20]){
            {8., 6., 6., 1., 4., 2., 7., 6., 5., 1., 8., 8., 2., 7., 7., 6., 8., 6., 5., 1.},
            {1., 7., 9., 1., 0., 9., 7., 4., 0., 0., 2., 9., 3., 9., 9., 1., 1., 5., 1., 3.},
            {6., 2., 7., 8., 8., 9., 7., 1., 1., 3., 1., 0., 5., 7., 4., 8., 4., 5., 4., 6.},
            {3., 9., 3., 9., 0., 2., 0., 4., 0., 5., 5., 8., 1., 7., 7., 7., 7., 5., 8., 1.},
            {4., 2., 1., 9., 6., 8., 9., 3., 9., 1., 2., 2., 0., 5., 3., 2., 1., 9., 6., 3.},
            {9., 6., 1., 3., 6., 3., 5., 3., 8., 8., 1., 4., 0., 7., 6., 4., 4., 9., 0., 2.},
            {5., 2., 4., 1., 5., 5., 3., 4., 7., 0., 4., 2., 4., 0., 1., 6., 9., 8., 4., 1.},
            {3., 5., 7., 9., 2., 8., 7., 0., 1., 2., 5., 6., 6., 6., 2., 4., 6., 0., 3., 2.},
            {5., 4., 4., 7., 3., 6., 4., 7., 7., 5., 0., 5., 5., 5., 3., 0., 2., 2., 1., 8.},
            {4., 9., 7., 0., 8., 8., 7., 6., 5., 9., 7., 0., 8., 5., 3., 6., 0., 4., 3., 2.}
        }, 2, (int[]){10, 20});
    tensor *bias = tensor_init_from_data(
        (float *)(float[10]){
            6., 8., 5., 2., 8., 7., 6., 2., 4., 9.
        }, 1, (int[]){10});
    tensor *expected = tensor_init_from_data(
        (float *)(float[10]){
            365., 367., 438., 376., 370., 381., 251., 369., 413., 417.
        }, 1, (int[]){10});

    tensor *dst = tensor_linear_new(weight, bias, src);
    tensor_check_equal(dst, expected, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

// Copy of the network from
// https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
void
test_cifar10() {
    tensor *conv1 = tensor_init(4, (int[]){6, 3, 5, 5});
    tensor_fill_rand_range(conv1, 10);

    tensor *conv1_bias = tensor_init(1, (int[]){6});
    tensor_fill_const(conv1_bias, 0);

    tensor *conv2 = tensor_init(4, (int[]){16, 6, 5, 5});
    tensor_fill_rand_range(conv2, 10);
    tensor *conv2_bias = tensor_init(1, (int[]){16});
    tensor_fill_const(conv1_bias, 0);

    tensor *fc1 = tensor_init(2, (int[]){120, 400});
    tensor_fill_rand_range(fc1, 10);
    tensor *fc1_bias = tensor_init(1, (int[]){120});

    tensor *fc2 = tensor_init(2, (int[]){84, 120});
    tensor_fill_rand_range(fc2, 10);
    tensor *fc2_bias = tensor_init(1, (int[]){84});

    tensor *fc3 = tensor_init(2, (int[]){10, 84});
    tensor_fill_rand_range(fc3, 10);
    tensor *fc3_bias = tensor_init(1, (int[]){10});

    tensor *x0 = tensor_init(3, (int[]){3, 32, 32});
    tensor_fill_rand_range(x0, 10);

    tensor *x1 = tensor_conv2d_new(conv1, conv1_bias, 1, 0, x0);
    tensor_unary(x1, x1, TENSOR_UNARY_OP_RELU);

    assert(x1->dims[0] == 6);
    assert(x1->dims[1] == 28);
    assert(x1->dims[2] == 28);

    tensor *x2 = tensor_max_pool2d_new(2, 2, 2, 0, x1);

    assert(x2->dims[0] == 6);
    assert(x2->dims[1] == 14);
    assert(x2->dims[2] == 14);

    tensor *x3 = tensor_conv2d_new(conv2, conv2_bias, 1, 0, x2);
    tensor_unary(x3, x3, TENSOR_UNARY_OP_RELU);
    assert(x3->dims[0] == 16);
    assert(x3->dims[1] == 10);
    assert(x3->dims[2] == 10);

    tensor *x4 = tensor_max_pool2d_new(2, 2, 2, 0, x3);
    assert(x4->dims[0] == 16);
    assert(x4->dims[1] == 5);
    assert(x4->dims[2] == 5);

    tensor_flatten(x4, 0);
    tensor_check_dims(x4, 1, (int[]){400});

    tensor *x5 = tensor_linear_new(fc1, fc1_bias, x4);
    tensor_unary(x5, x5, TENSOR_UNARY_OP_RELU);
    assert(tensor_check_dims(x5, 1, (int[]){120}));

    tensor *x6 = tensor_linear_new(fc2, fc2_bias, x5);
    tensor_unary(x6, x6, TENSOR_UNARY_OP_RELU);
    assert(x6->n_dims == 1);
    assert(x6->dims[0] == 84);

    tensor *x7 = tensor_linear_new(fc3, fc3_bias, x6);
    assert(x7->n_dims == 1);
    assert(x7->dims[0] == 10);

    tensor_free(conv1);
    tensor_free(conv1_bias);
    tensor_free(conv2);
    tensor_free(conv2_bias);
    tensor_free(fc1);
    tensor_free(fc1_bias);
    tensor_free(fc2);
    tensor_free(fc2_bias);
    tensor_free(fc3);
    tensor_free(fc3_bias);
    tensor_free(x0);
    tensor_free(x1);
    tensor_free(x2);
    tensor_free(x3);
    tensor_free(x4);
    tensor_free(x5);
    tensor_free(x6);
    tensor_free(x7);
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
    tensor_check_dims(x1, 3, (int[]){6, 28, 28});

    tensor *x2 = tensor_layer_apply_new(relu, x1);
    tensor_check_dims(x2, 3, (int[]){6, 28, 28});

    tensor *x3 = tensor_layer_apply_new(max_pool2d, x2);
    tensor_check_dims(x3, 3, (int[]){6, 14, 14});

    tensor *x4 = tensor_layer_apply_new(conv2, x3);
    tensor_check_dims(x4, 3, (int[]){16, 10, 10});

    tensor *x5 = tensor_layer_apply_new(max_pool2d, x4);
    tensor_check_dims(x5, 3, (int[]){16, 5, 5});

    tensor *x6 = tensor_layer_apply_new(flatten, x5);
    tensor_check_dims(x6, 1, (int[]){400});

    tensor *x7 = tensor_layer_apply_new(fc1, x6);
    tensor_check_dims(x7, 1, (int[]){120});

    tensor *x8 = tensor_layer_apply_new(fc2, x7);
    tensor_check_dims(x8, 1, (int[]){84});

    tensor *x9 = tensor_layer_apply_new(fc3, x8);
    tensor_check_dims(x9, 1, (int[]){10});

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
        tensor_layer_init_conv2d_from_data(2, 2, 2, 2, 0,
                                           (float *)conv1_weight, (float *)conv1_bias),
    };
    tensor_layer_stack *stack = tensor_layer_stack_init(
        ARRAY_SIZE(layers), layers,
        3, (int[]){2, 8, 8}
    );
    assert(tensor_n_elements(stack->src_buf) == 128);

    tensor *x0 = tensor_init_from_data((float *)x0_data, 3, (int[]){2, 8, 8});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 4, 4});
    tensor *x1 = tensor_layer_stack_apply_new(stack, x0);
    assert(x1);

    tensor_check_equal(x1, expected, LINALG_EPSILON);

    tensor_layer_stack_free(stack);
    tensor_free(x0);
    tensor_free(x1);
    tensor_free(expected);
}

void
test_softmax() {
    tensor *t = tensor_init_from_data((float *)(float[]){-1, 0, 3, 5},
                                      1, (int[]){4});
    tensor_unary(t, t, TENSOR_UNARY_OP_SOFTMAX);
    assert(approx_eq(t->data[0], 0.00216569646006f));
    assert(approx_eq(t->data[1], 0.00588697333334f));
    assert(approx_eq(t->data[2], 0.11824302025266f));
    assert(approx_eq(t->data[3], 0.87370430995393f));

    tensor_free(t);
}

void
test_transpose() {
    tensor *src = tensor_init_from_data((float *)mat_5x5_1,
                                        2, (int[]){5, 5});
    tensor *dst = tensor_init(2, (int[]){5, 5});
    tensor *dst_ref = tensor_init_from_data((float *)mat_5x5_2,
                                            2, (int[]){5, 5});
    tensor_transpose(src, dst);
    tensor_check_equal(dst, dst_ref, LINALG_EPSILON);

    tensor_free(dst);
    tensor_free(dst_ref);
    tensor_free(src);

    tensor *t1 = tensor_init_from_data((float *)mat_2x4,
                                       2, (int[]){2, 4});
    tensor *t2 = tensor_init(2, (int[]){4, 2});
    tensor *t2_ref = tensor_init_from_data((float *)mat_2x4_t,
                                           2, (int[]){4, 2});
    tensor_transpose(t1, t2);
    tensor_check_equal(t2, t2_ref, LINALG_EPSILON);

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t2_ref);
}

void
test_too_big() {
    int SIZE = 1 << 17;
    tensor *mat = tensor_init(2, (int[]){SIZE, SIZE});
    assert(mat->error_code == TENSOR_ERR_TOO_BIG);
    tensor_free(mat);
}

void
test_random_filling() {
    // 4024mb
    int SIZE = 1 << 14;
    tensor *mat = tensor_init(2, (int[]){SIZE, SIZE});
    assert(mat);
    tensor_fill_rand_range(mat, 10000);
    int geq1 = 0;
    for (int i = 0; i < 10; i++) {
        float v = mat->data[i];
        if (v > 1.0f) {
            geq1++;
        }
        assert(v < 10000);
    }
    assert(geq1 > 0);
    tensor_free(mat);
}

void
test_scans() {
    tensor *src = tensor_init(1, dim_10);
    tensor *dst = tensor_init(1, dim_10);
    tensor *dst_ref = tensor_init_from_data(
        &arr_10_triangular_numbers[1], 1, dim_10);

    tensor_fill_range(src, 1.0);
    tensor_scan(src, dst, TENSOR_BINARY_OP_ADD, false, 0.0);
    tensor_check_equal(dst, dst_ref, LINALG_EPSILON);

    tensor_scan(src, dst, TENSOR_BINARY_OP_ADD, true, 0.0);
    memcpy(dst_ref->data, &arr_10_triangular_numbers[0],
           sizeof(float) * 10);
    tensor_check_equal(dst, dst_ref, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(dst_ref);
}

void
test_print_tensor() {
    tensor *src = tensor_init_from_data((float *)mat_5x5_1,
                                        3, (int[]){1, 5, 5});
    tensor_print(src, true, 3, 80, ", ");
    tensor_free(src);
}

int
main(int argc, char *argv[]) {
    rand_init(1234);
    if (argc == 1) {
        printf("Specify a PNG image.\n");
        return 1;
    }
    fname = argv[1];
    PRINT_RUN(test_from_png);
    PRINT_RUN(test_pick_channel);
    PRINT_RUN(test_conv2d);
    PRINT_RUN(test_conv2d_3);
    PRINT_RUN(test_conv2d_strided);
    PRINT_RUN(test_conv2d_padded);
    PRINT_RUN(test_conv2d_padded_2);
    PRINT_RUN(test_conv2d_2channels);
    PRINT_RUN(test_conv2d_2x2channels);
    PRINT_RUN(test_conv2d_uneven);
    PRINT_RUN(test_conv2d_uneven_strided);
    PRINT_RUN(test_conv2d_with_bias);
    PRINT_RUN(test_max_pool_1);
    PRINT_RUN(test_max_pool_strided);
    PRINT_RUN(test_max_pool_image);
    PRINT_RUN(test_relu);
    PRINT_RUN(test_linear);
    PRINT_RUN(test_linear_with_bias);
    PRINT_RUN(test_cifar10);
    PRINT_RUN(test_lenet_layers);
    PRINT_RUN(test_lenet_layer_stack);
    PRINT_RUN(test_lenet_layer_stack_apply_relu);
    PRINT_RUN(test_layer_stack_apply_conv2d);
    PRINT_RUN(test_layer_stack_apply_lenet);
    PRINT_RUN(test_softmax);
    PRINT_RUN(test_transpose);
    PRINT_RUN(test_too_big);
    PRINT_RUN(test_random_filling);
    PRINT_RUN(test_scans);
    PRINT_RUN(test_print_tensor);
}
