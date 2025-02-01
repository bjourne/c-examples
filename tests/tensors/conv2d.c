// Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/conv2d.h"
#include "tensors/multiply.h"
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

void
test_blur_image() {
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
    tensor *weight = tensor_init_4d(3, 3, 3, 3);
    tensor_copy_data(weight, weight_data);
    tensor *bias = tensor_init_1d(3);
    tensor_fill_const(bias, 0);

    tensor *t2 = tensor_conv2d_new(weight, bias, 1, 1, t1);

    assert(tensor_write_png(t2, "out_04.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);
    tensor_free(t2);
    tensor_free(t1);
    tensor_free(weight);
    tensor_free(bias);
#endif
}

void
test_no_padding() {
    float weight_data[1][1][2][2] = {
        {
            {
                {4, 0},
                {4, 3}
            }
        }
    };
    float exp_data[1][4][4] = {
        {
            {13, 28, 32, 24},
            {18, 32, 35,  7},
            {19, 21, 37, 22},
            { 8, 13, 27, 16}
        }
    };
    tensor *src = tensor_init_3d(1, 5, 5);
    tensor_copy_data(src, mat_5x5_1);

    tensor *weight = tensor_init_4d(1, 1, 2, 2);
    tensor_copy_data(weight, weight_data);
    tensor *bias = tensor_init_1d(1);
    tensor_fill_const(bias, 0);

    tensor *exp = tensor_init_3d(1, 4, 4);
    tensor_copy_data(exp, exp_data);
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);
    tensor_check_equal(exp, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(exp);
}

void
test_no_padding2() {
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
    tensor *src = tensor_init_3d(2, 3, 6);
    tensor_copy_data(src, src_data);

    tensor *weight = tensor_init_4d(1, 2, 1, 1);
    tensor_copy_data(weight, weight_data);
    tensor *bias = tensor_init_1d(1);
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_3d(1, 3, 6);
    tensor_copy_data(expected, expected_data);

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);
    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}


void
test_padding() {
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
    tensor *src = tensor_init_3d(1, 5, 5);
    tensor_copy_data(src, src_data);

    tensor *weight = tensor_init_4d(1, 1, 2, 2);
    tensor_copy_data(weight, weight_data);
    tensor *bias = tensor_init_1d(1);
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_3d(1, 6, 6);
    tensor_copy_data(expected, expected_data);
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 1, src);
    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_2x2channels() {
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
    tensor *bias = tensor_init_1d(2);
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
test_strides() {
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

    tensor *weight = tensor_init_4d(1, 1, 3, 3);
    tensor_copy_data(weight, weight_data);
    tensor *bias = tensor_init_1d(1);
    tensor_fill_const(bias, 0);
    tensor *expected = tensor_init_3d(1, 3, 3);
    tensor_copy_data(expected, expected_data);
    tensor *src = tensor_init_3d(1, 5, 5);
    tensor_copy_data(src, src_data);
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

    tensor *src = tensor_init_3d(1, 10, 3);
    tensor_copy_data(src, src_data);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 1, 3, 3});
    tensor *bias = tensor_init_1d(1);
    tensor_fill_const(bias, 0);
    tensor *dst = tensor_init_3d(1, 10, 3);
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
test_uneven() {
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
    tensor *src = tensor_init_3d(1, 2, 4);
    tensor_copy_data(src, src_data);
    tensor *weight = tensor_init_4d(2, 1, 2, 2);
    tensor_copy_data(weight, weight_data);
    tensor *expected = tensor_init_3d(2, 1, 3);
    tensor_copy_data(expected, expected_data);
    tensor *bias = tensor_init_1d(2);
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
test_uneven_strided() {
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
    tensor *src = tensor_init_3d(1, 2, 4);
    tensor_copy_data(src, src_data);
    tensor *dst = tensor_init_3d(2, 1, 2);
    tensor *weight = tensor_init_4d(2, 1, 2, 2);
    tensor_copy_data(weight, weight_data);
    tensor *bias = tensor_init_1d(2);
    tensor_fill_const(bias, 0);

    tensor *expected = tensor_init_3d(2, 1, 2);
    tensor_copy_data(expected, expected_data);

    tensor_conv2d(weight, bias, 2, 0, src, dst);

    tensor_check_equal(expected, dst, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_bias() {
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
    tensor *src = tensor_init_3d(2, 5, 5);
    tensor_copy_data(src, src_data);

    tensor *weight = tensor_init_4d(2, 2, 3, 3);
    tensor_copy_data(weight, weight_data);
    tensor *expected = tensor_init_3d(2, 3, 3);
    tensor_copy_data(expected, expected_data);
    tensor *bias = tensor_init_1d(2);
    tensor_copy_data(bias, bias_data);
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst, LINALG_EPSILON);


    tensor_free(src);
    tensor_free(dst);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

void
test_stride3_padding0() {
    float src_data[1][5][5] = {
        {
            {7, 5, 6, 1, 3},
            {9, 3, 3, 2, 8},
            {1, 1, 6, 1, 8},
            {1, 8, 2, 3, 3},
            {1, 0, 7, 2, 4}
        },
    };
    float exp_data[1][2][2] = {
        {
            {7, 1},
            {1, 3}
        }
    };

    tensor *src = tensor_init_3d(1, 5, 5);
    tensor_copy_data(src, src_data);
    tensor *exp = tensor_init_3d(1, 2, 2);
    tensor_copy_data(exp, exp_data);
    tensor *weight = tensor_init_4d(1, 1, 1, 1);
    tensor_fill_const(weight, 1.0);
    tensor *bias = tensor_init_1d(1);
    tensor_fill_const(bias, 0);
    tensor *dst = tensor_conv2d_new(weight, bias, 3, 0, src);
    tensor_check_equal(exp, dst, LINALG_EPSILON);
    tensor_free(src);
    tensor_free(exp);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(dst);
}

// Dunno if this works as I think...
int
simple_checksum(tensor *me) {
    int chk = 0;
    float *D = me->data;
    for (int i = 0; i < tensor_n_elements(me); i++) {
        chk = chk + (i % 2 == 0 ? 1 : -1) * D[i];
    }
    return chk;
}

void
test_im2col() {
    tensor *src = tensor_init_3d(5, 5, 1);
    tensor_fill_range(src, 0);

    tensor *dst = tensor_im2col_new(src, 1, 1, 1, 1, 0, 0);
    assert(dst);
    tensor_check_dims(dst, 5, 5, 5, 1, 1, 1);
    tensor_check_equal_contents(src, dst, LINALG_EPSILON);
    tensor_free(dst);

    // 2x2 kernel
    dst = tensor_im2col_new(src, 2, 2, 1, 1, 0, 0);
    tensor_check_dims(dst, 5, 4, 4, 2, 2, 1);
    assert(simple_checksum(dst) == -32);
    tensor_free(dst);

    // 3x3 kernel
    dst = tensor_im2col_new(src, 3, 3, 1, 1, 0, 0);
    tensor_check_dims(dst, 5, 3, 3, 3, 3, 1);
    assert(simple_checksum(dst) == 12);
    tensor_free(dst);

    // 4x4 kernel
    dst = tensor_im2col_new(src, 4, 4, 1, 1, 0, 0);
    assert(simple_checksum(dst) == -32);
    tensor_free(dst);
    tensor_free(src);
}

void
test_im2col_strided() {
    tensor *src = tensor_init_3d(5, 5, 1);
    tensor_fill_range(src, 0);

    // 1x1 kernel, 2x2 stride
    tensor *dst = tensor_im2col_new(src, 1, 1, 2, 2, 0, 0);
    tensor_check_dims(dst, 5, 3, 3, 1, 1, 1);
    assert(simple_checksum(dst) == 12);
    tensor_free(dst);

    tensor_free(src);
}

void
test_im2col_padded() {
    tensor *src = tensor_init_3d(5, 5, 1);
    tensor_fill_range(src, 0);

    // 1x1 kernel, 1x1 padding
    tensor *dst = tensor_im2col_new(src, 1, 1, 1, 1, 1, 1);
    tensor_check_dims(dst, 5, 7, 7, 1, 1, 1);
    assert(simple_checksum(dst) == 12);
    tensor_free(dst);
    tensor_free(src);

    src = tensor_init_3d(5, 5, 3);
    tensor_fill_range(src, 0);

    // 3x3 kernel, 1x1 padding
    dst = tensor_im2col_new(src, 3, 3, 1, 1, 1, 1);
    tensor_check_dims(dst, 5, 5, 5, 3, 3, 3);

    tensor_free(dst);
    tensor_free(src);
}

void
test_im2col_conv2d() {
    int fy_dim = 3;
    int fx_dim = 3;
    int iy_dim = 8;
    int ix_dim = 6;
    int ic_dim = 9;
    int dc_dim = 3;

    int pad_y = 1;
    int pad_x = 1;
    int stride_y = 1;
    int stride_x = 1;

    int dy_dim = tensor_padded_strided_dim(iy_dim, fy_dim, pad_y, stride_y);
    int dx_dim = tensor_padded_strided_dim(ix_dim, fx_dim, pad_x, stride_x);

    tensor *src0 = tensor_init_3d(ic_dim, iy_dim, ix_dim);
    tensor_fill_range(src0, 0);

    tensor *weight0 = tensor_init_4d(dc_dim, ic_dim, fy_dim, fx_dim);
    tensor_fill_range(weight0, -20);
    tensor *bias = tensor_init_1d(dc_dim);
    tensor_fill_const(bias, 0);

    tensor *dst0 = tensor_conv2d_new(weight0, bias, stride_x, pad_x, src0);
    tensor_check_dims(dst0, 3, dc_dim, dy_dim, dx_dim);

    // Move channels to back
    tensor *src1 = tensor_permute_dims_new(src0, (int[]){1, 2, 0});
    tensor *weight1 = tensor_permute_dims_new(weight0, (int[]){2, 3, 1, 0});

    // Perform im2col
    tensor *src1_cols = tensor_im2col_new(
        src1,
        fy_dim, fx_dim,
        stride_y, stride_x,
        pad_y, pad_x
    );

    // Merge dims
    int col_dims[] = {dy_dim * dx_dim, fy_dim * fx_dim * ic_dim};
    int weight_dims[] = {fy_dim * fx_dim * ic_dim, dc_dim};
    tensor_set_dims(src1_cols, 2, col_dims);
    tensor_set_dims(weight1, 2, weight_dims);
    tensor *dst1_cols = tensor_multiply_new(src1_cols, weight1);

    // Move channels to front
    tensor *dst1 = tensor_permute_dims_new(dst1_cols, (int[]){1, 0});
    tensor_set_dims(dst1, 3, (int[]){dc_dim, dy_dim, dx_dim});
    tensor_print(dst0, true, 0, 160, " ");
    tensor_print(dst1, true, 0, 160, " ");
    tensor_check_equal(dst0, dst1, LINALG_EPSILON);

    tensor_free(src0);
    tensor_free(src1);
    tensor_free(src1_cols);
    tensor_free(dst0);
    tensor_free(dst1);
    tensor_free(dst1_cols);
    tensor_free(weight0);
    tensor_free(weight1);

    tensor_free(bias);
}

void
perf_im2col() {
    int stride_y = 1;
    int stride_x = 1;
    int pad_y = 1;
    int pad_x = 1;
    int sy_dim = 64;
    int sx_dim = 64;
    int sc_dim = 128;
    int fy_dim = 3;
    int fx_dim = 3;

    tensor *src = tensor_init_3d(sy_dim, sx_dim, 128);
    int dy_dim = tensor_padded_strided_dim(sy_dim, fy_dim, pad_y, stride_y);
    int dx_dim = tensor_padded_strided_dim(sx_dim, fx_dim, pad_x, stride_x);

    tensor *dst = tensor_init(5, (int[]){
        dy_dim, dx_dim, fy_dim, fx_dim, sc_dim
    });

    uint64_t bef = nano_count();
    for (int i = 0; i < 5000; i++) {
        tensor_im2col(src, dst, stride_y, stride_x, pad_y, pad_x);
    }
    printf("%.3lfs\n", nanos_to_secs(nano_count() - bef));

    tensor_free(src);
    tensor_free(dst);
}

// Copy of the network from
// https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
void
test_cifar10() {
    tensor *conv1 = tensor_init_4d(6, 3, 5, 5);
    tensor_fill_rand_range(conv1, 10);

    tensor *conv1_bias = tensor_init_1d(6);
    tensor_fill_const(conv1_bias, 0);

    tensor *conv2 = tensor_init_4d(16, 6, 5, 5);
    tensor_fill_rand_range(conv2, 10);
    tensor *conv2_bias = tensor_init_1d(16);
    tensor_fill_const(conv1_bias, 0);

    tensor *fc1 = tensor_init_2d(120, 400);
    tensor_fill_rand_range(fc1, 10);
    tensor *fc1_bias = tensor_init_1d(120);

    tensor *fc2 = tensor_init_2d(84, 120);
    tensor_fill_rand_range(fc2, 10);
    tensor *fc2_bias = tensor_init_1d(84);

    tensor *fc3 = tensor_init_2d(10, 84);
    tensor_fill_rand_range(fc3, 10);
    tensor *fc3_bias = tensor_init_1d(10);

    tensor *x0 = tensor_init_3d(3, 32, 32);
    tensor_fill_rand_range(x0, 10);

    tensor *x1 = tensor_conv2d_new(conv1, conv1_bias, 1, 0, x0);
    tensor_unary(x1, x1, TENSOR_UNARY_OP_MAX, 0);
    tensor_check_dims(x1, 3, 6, 28, 28);

    tensor *x2 = tensor_max_pool2d_new(2, 2, 2, 0, x1);
    tensor_check_dims(x2, 3, 6, 14, 14);

    tensor *x3 = tensor_conv2d_new(conv2, conv2_bias, 1, 0, x2);
    tensor_unary(x3, x3, TENSOR_UNARY_OP_MAX, 0);
    tensor_check_dims(x3, 3, 16, 10, 10);

    tensor *x4 = tensor_max_pool2d_new(2, 2, 2, 0, x3);
    tensor_check_dims(x4, 3, 16, 5, 5);
    tensor_flatten(x4, 0);
    tensor_check_dims(x4, 1, 400);

    tensor *x5 = tensor_linear_new(fc1, fc1_bias, x4);
    tensor_unary(x5, x5, TENSOR_UNARY_OP_MAX, 0);
    tensor_check_dims(x5, 1, 120);

    tensor *x6 = tensor_linear_new(fc2, fc2_bias, x5);
    tensor_unary(x6, x6, TENSOR_UNARY_OP_MAX, 0);
    tensor_check_dims(x6, 1, 84);

    tensor *x7 = tensor_linear_new(fc3, fc3_bias, x6);
    tensor_check_dims(x7, 1, 10);

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

int
main(int argc, char *argv[]) {
    rand_init(1234);
    if (argc == 1) {
        printf("Specify a PNG image.\n");
        return 1;
    }
    fname = argv[1];
    PRINT_RUN(test_blur_image);
    PRINT_RUN(test_no_padding);
    PRINT_RUN(test_no_padding2);
    PRINT_RUN(test_2x2channels);

    PRINT_RUN(test_conv2d_3);
    PRINT_RUN(test_uneven);
    PRINT_RUN(test_uneven_strided);
    PRINT_RUN(test_bias);

    //Im2Col
    PRINT_RUN(test_im2col);
    PRINT_RUN(test_im2col_strided);
    PRINT_RUN(test_im2col_padded);
    PRINT_RUN(test_im2col_conv2d);

    // Cifar10
    PRINT_RUN(test_cifar10);

    // Strides
    PRINT_RUN(test_strides);
    PRINT_RUN(test_stride3_padding0);

    // Padding
    PRINT_RUN(test_padding);

    perf_im2col();
}
