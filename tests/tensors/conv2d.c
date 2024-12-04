// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
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

    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 10, 3});
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
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 2, 4});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){2, 1, 2, 2});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 1, 3});
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
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, (int[]){1, 2, 4});
    tensor *dst = tensor_init_3d(2, 1, 2);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){2, 1, 2, 2});
    tensor *bias = tensor_init_1d(2);
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
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){2, 3, 3});
    tensor *bias = tensor_init_from_data((float *)bias_data,
                                         1, (int[]){2});
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
    tensor_fill_const(bias, 0.0);
    tensor *dst = tensor_conv2d_new(weight, bias, 3, 0, src);
    tensor_check_equal(exp, dst, LINALG_EPSILON);
    tensor_free(src);
    tensor_free(exp);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(dst);
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
    PRINT_RUN(test_strides);
    PRINT_RUN(test_conv2d_3);
    PRINT_RUN(test_uneven);
    PRINT_RUN(test_uneven_strided);
    PRINT_RUN(test_bias);
    PRINT_RUN(test_stride3_padding0);

    // With padding
    PRINT_RUN(test_padding);
}
