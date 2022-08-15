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

    int height = t1->dims[1];
    int width = t1->dims[2];
    tensor *t2 = tensor_init(3, 3, height, width);

    float w = 1/8.0;

    // Blur red and green channels and filter blue channel.
    float kernel_data[3][3][3][3] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 3, 3, 3, 3);
    tensor_conv2d(t1, kernel, t2, 1, 1);

    assert(tensor_write_png(t2, "out_04.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);
    tensor_free(t2);
    tensor_free(t1);
    tensor_free(kernel);
}

void
test_convolve_padded() {
    float src_data[1][5][5] = {
        {
            {0, 1, 4, 3, 2},
            {1, 3, 4, 0, 4},
            {2, 2, 4, 1, 1},
            {2, 1, 3, 3, 2},
            {0, 0, 3, 1, 0}
        }
    };
    float kernel_data[1][1][2][2] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 1, 1, 2, 2);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 4, 4);
    tensor *dst = tensor_init(3, 1, 4, 4);

    tensor_conv2d(src, kernel, dst, 1, 0);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}

void
test_convolve_padded_2() {
    float src_data[1][5][5] = {
        {
            {2, 3, 1, 0, 1},
            {2, 0, 3, 0, 4},
            {2, 0, 1, 3, 1},
            {4, 1, 2, 3, 1},
            {3, 4, 1, 4, 4}
        }
    };
    float kernel_data[1][1][2][2] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 1, 1, 2, 2);
    tensor *dst = tensor_init(3, 1, 6, 6);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 6, 6);

    tensor_conv2d(src, kernel, dst, 1, 1);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}

void
test_convolve_2channels() {
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
    float kernel_data[1][2][1][1] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 1, 2, 1, 1);
    tensor *dst = tensor_init(3, 1, 3, 6);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 3, 6);

    tensor_conv2d(src, kernel, dst, 1, 0);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}

void
test_convolve_2x2channels() {
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
    float kernel_data[2][2][1][1] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 2, 2, 1, 1);
    tensor *dst = tensor_init(3, 2, 2, 4);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 2, 2, 4);

    tensor_conv2d(src, kernel, dst, 1, 0);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}

void
test_convolve_strided() {
    float src_data[1][5][5] = {
        {
            {2, 4, 3, 3, 2},
            {4, 1, 3, 3, 4},
            {3, 1, 2, 2, 1},
            {3, 0, 2, 0, 0},
            {0, 2, 1, 0, 1}
        }
    };
    float kernel_data[1][1][3][3] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 1, 1,  3, 3);
    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 3, 3);
    tensor *dst = tensor_init(3, 1, 3, 3);

    tensor_conv2d(src, kernel, dst, 2, 1);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}


void
test_convolve_3() {
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
    float kernel_data[1][1][3][3] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 1, 1, 3, 3);
    tensor *dst = tensor_init(3, 1, 10, 3);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 1, 10, 3);

    tensor_conv2d(src, kernel, dst, 1, 1);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}

void
test_convolve_uneven() {
    float src_data[1][2][4] = {
        {
            {4, 1, 4, 0},
            {0, 0, 0, 2}
        }
    };
    float kernel_data[2][1][2][2] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 2, 1, 2, 2);
    tensor *dst = tensor_init(3, 2, 1, 3);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 2, 1, 3);

    tensor_conv2d(src, kernel, dst, 1, 0);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
    tensor_free(expected);
}

void
test_convolve_uneven_strided() {
    float src_data[1][2][4] = {
        {
            {3, 0, 3, 1},
            {3, 4, 2, 2}
        }
    };
    float kernel_data[2][1][2][2] = {
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
    tensor *kernel = tensor_init_from_data((float *)kernel_data,
                                               4, 2, 1, 2, 2);
    tensor *dst = tensor_init(3, 2, 1, 2);

    tensor *expected = tensor_init_from_data((float *)expected_data,
                                                 3, 2, 1, 2);

    tensor_conv2d(src, kernel, dst, 2, 0);

    tensor_check_equal(expected, dst);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(kernel);
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



int
main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("Specify a PNG image.\n");
        return 1;
    }
    fname = argv[1];
    PRINT_RUN(test_from_png);
    PRINT_RUN(test_pick_channel);
    PRINT_RUN(test_conv2d);
    PRINT_RUN(test_convolve_3);
    PRINT_RUN(test_convolve_strided);
    PRINT_RUN(test_convolve_padded);
    PRINT_RUN(test_convolve_padded_2);
    PRINT_RUN(test_convolve_2channels);
    PRINT_RUN(test_convolve_2x2channels);
    PRINT_RUN(test_convolve_uneven);
    PRINT_RUN(test_convolve_uneven_strided);
    PRINT_RUN(test_max_pool_1);
    PRINT_RUN(test_max_pool_image);
    PRINT_RUN(test_relu);
}
