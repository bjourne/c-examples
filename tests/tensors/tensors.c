// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"

char *fname = NULL;

void
test_from_png() {
    tensor *t = tensor_read_png(fname);
    assert(t);
    assert(t->error_code == TENSOR_ERR_NONE);

    assert(tensor_write_png(t, "out_01.png"));
    assert(t->error_code == TENSOR_ERR_NONE);

    tensor_free(t);
}

void
test_pick_channel() {
    tensor *t1 = tensor_read_png(fname);
    assert(t1);
    assert(t1->error_code == TENSOR_ERR_NONE);

    int height = t1->dims[1];
    int width = t1->dims[2];

    tensor *t2 = tensor_init(3, 3, height, width);
    tensor_fill(t2, 0.0);
    for (int c = 0; c < 1; c++) {
        float *dst = &t2->data[c * height * width];
        memcpy(dst, t1->data, height * width * sizeof(float));
    }
    assert(tensor_write_png(t2, "out_02.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);

    tensor_free(t1);
    tensor_free(t2);
}

void
test_conv2d() {
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
                                           4, 3, 3, 3, 3);
    tensor *bias = tensor_init_filled(0, 1, 3);
    tensor *t2 = tensor_conv2d_new(weight, bias, 1, 1, t1);

    assert(tensor_write_png(t2, "out_04.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);
    tensor_free(t2);
    tensor_free(t1);
    tensor_free(weight);
}

void
test_conv2d_padded() {
    float src_data[1][5][5] = {
        {
            {0, 1, 4, 3, 2},
            {1, 3, 4, 0, 4},
            {2, 2, 4, 1, 1},
            {2, 1, 3, 3, 2},
            {0, 0, 3, 1, 0}
        }
    };
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
    tensor *src = tensor_init_from_data((float *)src_data,
                                        3, 1, 5, 5);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 1, 1, 2, 2);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 1, 4, 4);
    tensor *bias = tensor_init_filled(0, 1, 1);

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst);

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
                                        3, 1, 5, 5);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 1, 1, 2, 2);
    tensor *bias = tensor_init_filled(0, 1, 1);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 1, 6, 6);

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 1, src);

    tensor_check_equal(expected, dst);

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
                                            3, 2, 3, 6);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                               4, 1, 2, 1, 1);
    tensor *bias = tensor_init_filled(0, 1, 1);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 3, 6);

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst);

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
                                        3, 2, 2, 4);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 2, 2, 1, 1);
    tensor *bias = tensor_init_filled(0, 1, 2);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 2, 2, 4);

    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst);

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
    tensor *src = tensor_init_from_data((float *)src_data,
                                            3, 1, 5, 5);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 1, 1, 3, 3);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 3, 3);
    tensor *bias = tensor_init_filled(0, 1, 1);

    tensor *dst = tensor_conv2d_new(weight, bias, 2, 1, src);

    tensor_check_equal(expected, dst);

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
                                            3, 1, 10, 3);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                               4, 1, 1, 3, 3);
    tensor *bias = tensor_init_filled(0, 1, 1);
    tensor *dst = tensor_init(3, 1, 10, 3);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 1, 10, 3);

    tensor_conv2d(weight, bias, 1, 1, src, dst);

    tensor_check_equal(expected, dst);

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
                                        3, 1, 2, 4);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 2, 1, 2, 2);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 2, 1, 3);
    tensor *bias = tensor_init_filled(0, 1, 2);
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst);

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
                                        3, 1, 2, 4);
    tensor *dst = tensor_init(3, 2, 1, 2);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 2, 1, 2, 2);
    tensor *bias = tensor_init_filled(0, 1, 2);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 2, 1, 2);

    tensor_conv2d(weight, bias, 2, 0, src, dst);

    tensor_check_equal(expected, dst);

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
                                        3, 2, 5, 5);
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, 2, 2, 3, 3);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 2, 3, 3);
    tensor *bias = tensor_init_from_data((float *)bias_data, 1, 2);
    tensor *dst = tensor_conv2d_new(weight, bias, 1, 0, src);

    tensor_check_equal(expected, dst);


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
                                        3, 2, 5, 5);
    tensor *expected1 = tensor_init_from_data((float *)expected1_data,
                                              3, 2, 4, 4);
    tensor *expected2 = tensor_init_from_data((float *)expected2_data,
                                              3, 2, 1, 5);
    tensor *dst1 = tensor_init(3, 2, 4, 4);
    tensor *dst2 = tensor_init(3, 2, 1, 5);

    tensor_max_pool2d(src, 2, 2, dst1, 1, 0);
    tensor_check_equal(expected1, dst1);

    tensor_max_pool2d(src, 5, 1, dst2, 1, 0);
    tensor_check_equal(expected2, dst2);

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
                                        3, 2, 5, 5);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, 2, 2, 2);
    tensor *dst = tensor_max_pool2d_new(src, 2, 2, 2, 0);
    tensor_check_equal(expected, dst);
    tensor_free(src);
    tensor_free(expected);
}

void
test_max_pool_image() {
    tensor *src = tensor_read_png(fname);
    assert(src);
    assert(src->error_code == TENSOR_ERR_NONE);
    for (int s = 1; s < 5; s++) {
        tensor *dst = tensor_max_pool2d_new(src, s, s, 1, 0);

        char fname_fmt[128];
        sprintf(fname_fmt, "out-max_pool-%02d.png", s - 1);
        assert(tensor_write_png(dst, fname_fmt));
        assert(dst->error_code == TENSOR_ERR_NONE);
        tensor_free(dst);
    }
    tensor_free(src);
}

void
test_relu() {
    tensor *src = tensor_init_from_data(
        (float *)(float[]){-4.0, 0.0, -20.0, 3.0, 2.0}, 1, 5);
    tensor *expected = tensor_init_from_data(
        (float *)(float[]){0.0, 0.0, 0.0, 3.0, 2.0}, 1, 5);

    tensor_relu(src);
    tensor_check_equal(src, expected);

    tensor_free(src);
    tensor_free(expected);
}

void
test_linear() {
    tensor *src = tensor_init_from_data(
        (float *)(float[]){
            0, 2, 4, 5, 0, 1, 4, 4
        }, 1, 8);
    tensor *weight = tensor_init_from_data(
        (float *)(float[4][8]){
            {5, 6, 7, 6, 9, 3, 1, 9},
            {4, 9, 5, 0, 9, 9, 8, 7},
            {6, 8, 7, 4, 1, 9, 3, 1},
            {4, 1, 4, 8, 2, 0, 1, 1}
        }, 2, 4, 8);
    tensor *bias = tensor_init_from_data(
        (float *)(float[]){0, 0, 0, 0},
        1,  4);

    tensor *dst = tensor_init(1, 4);
    tensor *expected = tensor_init_from_data(
        (float *)(float[]){113, 107, 89, 66}, 1, 4);

    tensor_linear(src, weight, bias, dst);
    tensor_check_equal(dst, expected);

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
        }, 1, 20);
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
        }, 2, 10, 20);
    tensor *bias = tensor_init_from_data(
        (float *)(float[10]){
            6., 8., 5., 2., 8., 7., 6., 2., 4., 9.
        }, 1, 10);
    tensor *expected = tensor_init_from_data(
        (float *)(float[10]){
            365., 367., 438., 376., 370., 381., 251., 369., 413., 417.
        }, 1, 10);

    tensor *dst  = tensor_linear_new(src, weight, bias);
    tensor_check_equal(dst, expected);

    tensor_free(src);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(expected);
}

// Copy of the network from
// https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
void
test_cifar10() {
    tensor *conv1 = tensor_init(4, 6, 3, 5, 5);
    tensor_randrange(conv1, 10);
    tensor *conv1_bias = tensor_init_filled(0, 1, 6);

    tensor *conv2 = tensor_init(4, 16, 6, 5, 5);
    tensor_randrange(conv2, 10);
    tensor *conv2_bias = tensor_init_filled(0, 1, 16);

    tensor *fc1 = tensor_init(2, 120, 400);
    tensor_randrange(fc1, 10);
    tensor *fc1_bias = tensor_init(1, 120);

    tensor *fc2 = tensor_init(2, 84, 120);
    tensor_randrange(fc2, 10);
    tensor *fc2_bias = tensor_init(1, 84);

    tensor *fc3 = tensor_init(2, 10, 84);
    tensor_randrange(fc3, 10);
    tensor *fc3_bias = tensor_init(1, 10);

    tensor *x0 = tensor_init(3, 3, 32, 32);
    tensor_randrange(x0, 10);

    tensor *x1 = tensor_conv2d_new(conv1, conv1_bias, 1, 0, x0);
    tensor_relu(x1);

    assert(x1->dims[0] == 6);
    assert(x1->dims[1] == 28);
    assert(x1->dims[2] == 28);

    tensor *x2 = tensor_max_pool2d_new(x1, 2, 2, 2, 0);

    assert(x2->dims[0] == 6);
    assert(x2->dims[1] == 14);
    assert(x2->dims[2] == 14);

    tensor *x3 = tensor_conv2d_new(conv2, conv2_bias, 1, 0, x2);
    tensor_relu(x3);
    assert(x3->dims[0] == 16);
    assert(x3->dims[1] == 10);
    assert(x3->dims[2] == 10);

    tensor *x4 = tensor_max_pool2d_new(x3, 2, 2, 2, 0);
    assert(x4->dims[0] == 16);
    assert(x4->dims[1] == 5);
    assert(x4->dims[2] == 5);

    tensor_flatten(x4, 0);
    assert(x4->n_dims == 1);
    assert(x4->dims[0] == 400);

    tensor *x5 = tensor_linear_new(x4, fc1, fc1_bias);
    tensor_relu(x5);
    assert(x5->n_dims == 1);
    assert(x5->dims[0] == 120);

    tensor *x6 = tensor_linear_new(x5, fc2, fc2_bias);
    tensor_relu(x6);
    assert(x6->n_dims == 1);
    assert(x6->dims[0] == 84);

    tensor *x7 = tensor_linear_new(x6, fc3, fc3_bias);
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
}
