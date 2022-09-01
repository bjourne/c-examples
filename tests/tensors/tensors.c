// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
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

    tensor *t2 = tensor_init(3, (int[]){3, height, width});
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
                                           4, (int[]){3, 3, 3, 3});
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
                                        3, (int[]){1, 5, 5});
    tensor *weight = tensor_init_from_data((float *)weight_data,
                                           4, (int[]){1, 1, 2, 2});
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                             3, (int[]){1, 4, 4});
    tensor *bias = tensor_init_filled(0, 1, 1);

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
    tensor *bias = tensor_init_filled(0, 1, 1);

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
    tensor *bias = tensor_init_filled(0, 1, 1);

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
    tensor *bias = tensor_init_filled(0, 1, 2);

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
    tensor *bias = tensor_init_filled(0, 1, 1);
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
    tensor *bias = tensor_init_filled(0, 1, 1);
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
    tensor *bias = tensor_init_filled(0, 1, 2);
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
    tensor *bias = tensor_init_filled(0, 1, 2);
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
}

void
test_relu() {
    tensor *src = tensor_init_from_data(
        (float *)(float[]){-4.0, 0.0, -20.0, 3.0, 2.0},
        1, (int[]){5});
    tensor *expected = tensor_init_from_data(
        (float *)(float[]){0.0, 0.0, 0.0, 3.0, 2.0},
        1, (int[]){5});

    tensor_relu(src);
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
    tensor_randrange(conv1, 10);
    tensor *conv1_bias = tensor_init_filled(0, 1, 6);

    tensor *conv2 = tensor_init(4, (int[]){16, 6, 5, 5});
    tensor_randrange(conv2, 10);
    tensor *conv2_bias = tensor_init_filled(0, 1, 16);

    tensor *fc1 = tensor_init(2, (int[]){120, 400});
    tensor_randrange(fc1, 10);
    tensor *fc1_bias = tensor_init(1, (int[]){120});

    tensor *fc2 = tensor_init(2, (int[]){84, 120});
    tensor_randrange(fc2, 10);
    tensor *fc2_bias = tensor_init(1, (int[]){84});

    tensor *fc3 = tensor_init(2, (int[]){10, 84});
    tensor_randrange(fc3, 10);
    tensor *fc3_bias = tensor_init(1, (int[]){10});

    tensor *x0 = tensor_init(3, (int[]){3, 32, 32});
    tensor_randrange(x0, 10);

    tensor *x1 = tensor_conv2d_new(conv1, conv1_bias, 1, 0, x0);
    tensor_relu(x1);

    assert(x1->dims[0] == 6);
    assert(x1->dims[1] == 28);
    assert(x1->dims[2] == 28);

    tensor *x2 = tensor_max_pool2d_new(2, 2, 2, 0, x1);

    assert(x2->dims[0] == 6);
    assert(x2->dims[1] == 14);
    assert(x2->dims[2] == 14);

    tensor *x3 = tensor_conv2d_new(conv2, conv2_bias, 1, 0, x2);
    tensor_relu(x3);
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
    tensor_relu(x5);
    assert(tensor_check_dims(x5, 1, (int[]){120}));

    tensor *x6 = tensor_linear_new(fc2, fc2_bias, x5);
    tensor_relu(x6);
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
    tensor_softmax(t);
    assert(approx_eq(t->data[0], 0.00216569646006f));
    assert(approx_eq(t->data[1], 0.00588697333334f));
    assert(approx_eq(t->data[2], 0.11824302025266f));
    assert(approx_eq(t->data[3], 0.87370430995393f));

    tensor_free(t);
}

void
test_multiply() {
    tensor *as[] = {
        tensor_init_from_data(
            (float *)(float[]){-1, 0, 3, 5},
            2, (int[]){2, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){15, 2, 3, 4, 5, 6, 7, 8, 9},
            2, (int[]){3, 3}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){4, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){2, 4}
        )
    };
    tensor *bs[] = {
        tensor_init_from_data(
            (float *)(float[]){-1, 0, 3, 5},
            2, (int[]){2, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){10, 11, 12, 13, 14, 15, 16, 17, 18},
            2, (int[]){3, 3}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){2, 4}
        ),
        tensor_init_from_data(
            (float *)(float[]){0, 1, 2, 3, 4, 5, 6, 7},
            2, (int[]){4, 2}
        ),
    };
    tensor *cs[] = {
        tensor_init(2, (int[]){2, 2}),
        tensor_init(2, (int[]){3, 3}),
        tensor_init(2, (int[]){4, 4}),
        tensor_init(2, (int[]){2, 2})
    };
    tensor *c_exps[] = {
        tensor_init_from_data(
            (float *)(float[]){1, 0, 12, 25},
            2, (int[]){2, 2}
        ),
        tensor_init_from_data(
            (float *)(float[]){224, 244, 264, 201, 216, 231, 318, 342, 366},
            2, (int[]){3, 3}
        ),
        tensor_init_from_data(
            (float *)(float[]){
                4, 5, 6, 7,
                12, 17, 22, 27,
                20, 29, 38, 47,
                28, 41, 54, 67
            },
            2, (int[]){4, 4}
        ),
        tensor_init_from_data(
            (float *)(float[]){28, 34, 76, 98},
            2, (int[]){2, 2}
        )
    };
    for (int i = 0; i < ARRAY_SIZE(as); i++) {
        tensor *a = as[i];
        tensor *b = bs[i];
        tensor *c = cs[i];
        tensor *c_exp = c_exps[i];
        tensor_multiply(a, b, c);
        tensor_check_equal(c, c_exp, LINALG_EPSILON);
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(c_exp);
    }
}

void
test_dct() {
    int dims[][2] = {
        {8, 8},
        {16, 16},
        {8, 16},
    };
    float tot[] = {2040, 4080, 2885};
    for (int i = 0; i < ARRAY_SIZE(dims); i++) {
        int height = dims[i][0];
        int width = dims[i][1];
        tensor *a = tensor_init(2, dims[i]);
        tensor *b = tensor_init(2, dims[i]);
        tensor *c = tensor_init(2, dims[i]);
        tensor_fill(a, 255.0);
        tensor_dct2d_rect(a, b, 0, 0, height, width);
        assert(approx_eq2(b->data[0], tot[i], 0.005));
        tensor_idct2d(b, c);
        tensor_check_equal(a, c, 0.001);
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
    }
}

void
test_dct2() {
    float image_data[32][32] = {
        {83,  86,  77,  15,  93,  35,  86,  92,  49,  21,  62,  27,  90,  59,  63,  26,
         40,  26,  72,  36,  11,  68,  67,  29,  82,  30,  62,  23,  67,  35,  29,   2},
        { 22,  58,  69,  67,  93,  56,  11,  42,  29,  73,  21,  19,  84,  37,  98,  24,
          15,  70,  13,  26,  91,  80,  56,  73,  62,  70,  96,  81,   5,  25,  84,  27},
        { 36,   5,  46,  29,  13,  57,  24,  95,  82,  45,  14,  67,  34,  64,  43,  50,
          87,   8,  76,  78,  88,  84,   3,  51,  54,  99,  32,  60,  76,  68,  39,  12},
        { 26,  86,  94,  39,  95,  70,  34,  78,  67,   1,  97,   2,  17,  92,  52,  56,
          1,  80,  86,  41,  65,  89,  44,  19,  40,  29,  31,  17,  97,  71,  81,  75},
        {  9,  27,  67,  56,  97,  53,  86,  65,   6,  83,  19,  24,  28,  71,  32,  29,
           3,  19,  70,  68,   8,  15,  40,  49,  96,  23,  18,  45,  46,  51,  21,  55},
        { 79,  88,  64,  28,  41,  50,  93,   0,  34,  64,  24,  14,  87,  56,  43,  91,
          27,  65,  59,  36,  32,  51,  37,  28,  75,   7,  74,  21,  58,  95,  29,  37},
        { 35,  93,  18,  28,  43,  11,  28,  29,  76,   4,  43,  63,  13,  38,   6,  40,
          4,  18,  28,  88,  69,  17,  17,  96,  24,  43,  70,  83,  90,  99,  72,  25},
        { 44,  90,   5,  39,  54,  86,  69,  82,  42,  64,  97,   7,  55,   4,  48,  11,
          22,  28,  99,  43,  46,  68,  40,  22,  11,  10,   5,   1,  61,  30,  78,   5},
        { 20,  36,  44,  26,  22,  65,   8,  16,  82,  58,  24,  37,  62,  24,   0,  36,
          52,  99,  79,  50,  68,  71,  73,  31,  81,  30,  33,  94,  60,  63,  99,  81},
        { 99,  96,  59,  73,  13,  68,  90,  95,  26,  66,  84,  40,  90,  84,  76,  42,
          36,   7,  45,  56,  79,  18,  87,  12,  48,  72,  59,   9,  36,  10,  42,  87},
        {  6,   1,  13,  72,  21,  55,  19,  99,  21,   4,  39,  11,  40,  67,   5,  28,
           27,  50,  84,  58,  20,  24,  22,  69,  96,  81,  30,  84,  92,  72,  72,  50},
        { 25,  85,  22,  99,  40,  42,  98,  13,  98,  90,  24,  90,   9,  81,  19,  36,
          32,  55,  94,   4,  79,  69,  73,  76,  50,  55,  60,  42,  79,  84,  93,   5},
        { 21,  67,   4,  13,  61,  54,  26,  59,  44,   2,   2,   6,  84,  21,  42,  68,
          28,  89,  72,   8,  58,  98,  36,   8,  53,  48,   3,  33,  33,  48,  90,  54},
        { 67,  46,  68,  29,   0,  46,  88,  97,  49,  90,   3,  33,  63,  97,  53,  92,
          86,  25,  52,  96,  75,  88,  57,  29,  36,  60,  14,  21,  60,   4,  28,  27},
        { 50,  48,  56,   2,  94,  97,  99,  43,  39,   2,  28,   3,   0,  81,  47,  38,
          59,  51,  35,  34,  39,  92,  15,  27,   4,  29,  49,  64,  85,  29,  43,  35},
        { 77,   0,  38,  71,  49,  89,  67,  88,  92,  95,  43,  44,  29,  90,  82,  40,
          41,  69,  26,  32,  61,  42,  60,  17,  23,  61,  81,   9,  90,  25,  96,  67},
        { 77,  34,  90,  26,  24,  57,  14,  68,   5,  58,  12,  86,   0,  46,  26,  94,
          16,  52,  78,  29,  46,  90,  47,  70,  51,  80,  31,  93,  57,  27,  12,  86},
        { 14,  55,  12,  90,  12,  79,  10,  69,  89,  74,  55,  41,  20,  33,  87,  88,
          38,  66,  70,  84,  56,  17,   6,  60,  49,  37,   5,  59,  17,  18,  45,  83},
        { 73,  58,  73,  37,  89,  83,   7,  78,  57,  14,  71,  29,   0,  59,  18,  38,
          25,  88,  74,  33,  57,  81,  93,  58,  70,  99,  17,  39,  69,  63,  22,  94},
        { 73,  47,  31,  62,  82,  90,  92,  91,  57,  15,  21,  57,  74,  91,  47,  51,
          31,  21,  37,  40,  54,  30,  98,  25,  81,  16,  16,   2,  31,  39,  96,   4},
        { 38,  80,  18,  21,  70,  62,  12,  79,  77,  85,  36,   4,  76,  83,   7,  59,
          57,  44,  99,  11,  27,  50,  36,  60,  18,   5,  63,  49,  44,  11,   5,  34},
        { 91,  75,  55,  14,  89,  68,  93,  18,   5,  82,  22,  82,  17,  30,  93,  74,
          26,  93,  86,  53,  43,  74,  14,  13,  79,  77,  62,  75,  88,  19,  10,  32},
        { 94,  17,  46,  35,  37,  91,  53,  43,  73,  28,  25,  91,  10,  18,  17,  36,
          63,  55,  90,  58,  30,   4,  71,  61,  33,  85,  89,  73,   4,  51,   5,  50},
        { 68,   3,  85,   6,  95,  39,  49,  20,  67,  26,  63,  77,  96,  81,  65,  60,
          36,  55,  70,  18,  11,  42,  32,  96,  79,  21,  70,  84,  72,  27,  34,  40},
        { 83,  72,  98,  30,  63,  47,  50,  30,  73,  14,  59,  22,  47,  24,  82,  35,
          32,   4,  54,  43,  98,  86,  40,  78,  59,  62,  62,  83,  41,  48,  23,  24},
        { 72,  22,  54,  35,  21,  57,  65,  47,  71,  76,  69,  18,   1,   3,  53,  33,
          7,  59,  28,   6,  97,  20,  84,   8,  34,  98,  91,  76,  98,  15,  52,  71},
        { 89,  59,   6,  10,  16,  24,   9,  39,   0,  78,   9,  53,  81,  14,  38,  89,
          26,  67,  47,  23,  87,  31,  32,  22,  81,  75,  50,  79,  90,  54,  50,  31},
        { 13,  57,  94,  81,  81,   3,  20,  33,  82,  81,  87,  15,  96,  25,   4,  22,
          92,  51,  97,  32,  34,  81,   6,  15,  57,   8,  95,  99,  62,  97,  83,  76},
        { 54,  77,   9,  87,  32,  82,  21,  66,  63,  60,  82,  11,  85,  86,  85,  30,
          90,  83,  14,  76,  16,  20,  92,  25,  28,  39,  25,  90,  36,  60,  18,  43},
        { 37,  28,  82,  21,  10,  55,  88,  25,  15,  70,  37,  53,   8,  22,  83,  50,
          57,  97,  27,  26,  69,  71,  51,  49,  10,  28,  39,  98,  88,  10,  93,  77},
        { 90,  76,  99,  52,  31,  87,  77,  99,  57,  66,  52,  17,  41,  35,  68,  98,
          84,  95,  76,   5,  66,  28,  54,  28,   8,  93,  78,  97,  55,  72,  74,  45},
        {  0,  25,  97,  83,  12,  27,  82,  21,  93,  34,  39,  34,  21,  59,  85,  57,
           54,  61,  62,  72,  41,  16,  52,  50,  62,  82,  99,  17,  54,  73,  15,   6}
    };
    tensor *image = tensor_init_from_data((float *)image_data, 2, (int[]){32, 32});
    tensor *output = tensor_init(2, (int[]){32,  32});
    tensor_dct2d_blocked(image, output, 8, 8);
    assert(approx_eq2(output->data[   0], 433.62, 0.01));
    assert(approx_eq2(output->data[   1], -16.54, 0.01));
    assert(approx_eq2(output->data[1023], -49.20, 0.01));
    tensor_free(image);
    tensor_free(output);
}

void
test_multiply_big() {
    int dim = 1024;
    tensor *a = tensor_init(2, (int[]){dim, dim});
    tensor *b = tensor_init(2, (int[]){dim, dim});
    tensor *c = tensor_init(2, (int[]){dim, dim});
    tensor *c_exp = tensor_init(2, (int[]){dim, dim});

    tensor_randrange(a, 10.0);
    tensor_randrange(b, 10.0);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                float av = a->data[dim * i + k];
                float bv = b->data[dim * k + j];
                c_exp->data[dim * i + j] += av * bv;
            }
        }
    }
    tensor_multiply(a, b, c);
    tensor_check_equal(c, c_exp);


    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(c_exp);
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
    PRINT_RUN(test_multiply);
    PRINT_RUN(test_multiply_big);
    PRINT_RUN(test_dct);
    PRINT_RUN(test_dct2);

}
