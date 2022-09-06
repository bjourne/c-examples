// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"
#include "tensors/tensors-dct.h"

static float IMAGE_DATA8X8[8][8] = {
    {83, 86, 77, 15, 93, 35, 86, 92},
    {49, 21, 62, 27, 90, 59, 63, 26},
    {40, 26, 72, 36, 11, 68, 67, 29},
    {82, 30, 62, 23, 67, 35, 29,  2},
    {22, 58, 69, 67, 93, 56, 11, 42},
    {29, 73, 21, 19, 84, 37, 98, 24},
    {15, 70, 13, 26, 91, 80, 56, 73},
    {62, 70, 96, 81,  5, 25, 84, 27}
};

static float IMAGE_DATA32X32[32][32] = {
    {
        83, 86, 77, 15, 93, 35, 86, 92,
        49, 21, 62, 27, 90, 59, 63, 26,
        40, 26, 72, 36, 11, 68, 67, 29,
        82, 30, 62, 23, 67, 35, 29,  2
    },
    {
        22,  58,  69,  67,  93,  56,  11,  42,
        29,  73,  21,  19,  84,  37,  98,  24,
        15,  70,  13,  26,  91,  80,  56,  73,
        62,  70,  96,  81,   5,  25,  84,  27
    },
    {
        36,   5,  46,  29,  13,  57,  24,  95,
        82,  45,  14,  67,  34,  64,  43,  50,
        87,   8,  76,  78,  88,  84,   3,  51,
        54,  99,  32,  60,  76,  68,  39,  12
    },
    {
        26,  86,  94,  39,  95,  70,  34,  78,
        67,   1,  97,   2,  17,  92,  52,  56,
        1,  80,  86,  41,  65,  89,  44,  19,
        40,  29,  31,  17,  97,  71,  81,  75
    },
    {
        9,  27,  67,  56,  97,  53,  86,  65,
        6,  83,  19,  24,  28,  71,  32,  29,
        3,  19,  70,  68,   8,  15,  40,  49,
        96,  23,  18,  45,  46,  51,  21,  55
    },
    {
        79,  88,  64,  28,  41,  50,  93,   0,
        34,  64,  24,  14,  87,  56,  43,  91,
        27,  65,  59,  36,  32,  51,  37,  28,
        75,   7,  74,  21,  58,  95,  29,  37
    },
    {
        35,  93,  18,  28,  43,  11,  28,  29,
        76,   4,  43,  63,  13,  38,   6,  40,
        4,  18,  28,  88,  69,  17,  17,  96,
        24,  43,  70,  83,  90,  99,  72,  25
    },
    {
        44,  90,   5,  39,  54,  86,  69,  82,
        42,  64,  97,   7,  55,   4,  48,  11,
        22,  28,  99,  43,  46,  68,  40,  22,
        11,  10,   5,   1,  61,  30,  78,   5
    },
    {
        20,  36,  44,  26,  22,  65,   8,  16,
        82,  58,  24,  37,  62,  24,   0,  36,
        52,  99,  79,  50,  68,  71,  73,  31,
        81,  30,  33,  94,  60,  63,  99,  81
    },
    {
        99,  96,  59,  73,  13,  68,  90,  95,
        26,  66,  84,  40,  90,  84,  76,  42,
        36,   7,  45,  56,  79,  18,  87,  12,
        48,  72,  59,   9,  36,  10,  42,  87
    },
    {
        6,   1,  13,  72,  21,  55,  19,  99,
        21,   4,  39,  11,  40,  67,   5,  28,
        27,  50,  84,  58,  20,  24,  22,  69,
        96,  81,  30,  84,  92,  72,  72,  50
    },
    {
        25,  85,  22,  99,  40,  42,  98,  13,
        98,  90,  24,  90,   9,  81,  19,  36,
        32,  55,  94,   4,  79,  69,  73,  76,
        50,  55,  60,  42,  79,  84,  93,   5
    },
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
    tensor *image = tensor_init_from_data((float *)IMAGE_DATA32X32, 2, (int[]){32, 32});
    tensor *output = tensor_init(2, (int[]){32,  32});
    tensor_dct2d_blocks(image, output, 8, 8);
    assert(approx_eq2(output->data[   0], 433.62, 0.01));
    assert(approx_eq2(output->data[   1], -16.54, 0.01));
    assert(approx_eq2(output->data[1023], -49.20, 0.01));
    tensor_free(image);
    tensor_free(output);
}

void
test_dct_nonsquare() {
    float image_data[5][16] = {
        {0.96, 0.67, 0.89, 0.02, 0.69, 0.66, 0.73, 0.34,
         0.23, 0.59, 0.88, 0.28, 0.6 , 0.57, 0.96, 0.42},
        {0.29, 0.21, 0.19, 0.33, 0.17, 0.71, 0.47, 0.50,
         0.40, 0.55, 0.30, 0.62, 0.65, 0.72, 0.24, 0.12},
        {0.89, 0.53, 0.79, 0.16, 0.12, 0.21, 0.88, 0.26,
         0.87, 0.01, 0.13, 0.51, 0.56, 0.23, 0.51, 0.68},
        {0.14, 0.79, 0.45, 0.79, 0.91, 0.3 , 0.22, 0.09,
         0.62, 0.06, 0.13, 0.68, 0.7 , 0.28, 0.09, 0.68},
        {0.73, 0.06, 0.28, 0.47, 0.4 , 0.63, 0.72, 0.14,
         0.79, 0.01, 0.99, 0.72, 0.73, 0.56, 0.64, 0.96}
    };
    float ref_data[5][16] = {
        {
            4.37, -0.14,  0.31,  0.03, -0.06,  0.4 ,  0.2 , -0.2 ,
            0.15, -0.27,  0.06,  0.34,  0.14, -0.19,  0.02,  0.62
        },
        {
            0.05,  0.21, -0.09,  0.21,  0.2 ,  0.3 , -0.1 ,  0.24,
            -0.76, 0.11,  0.02, -0.1 , -0.07,  0.61,  0.16, -0.27
        },
        {
            0.52, -0.29,  0.03, -0.01, -0.15, -0.02,  0.53,  0.28,
            -0.39, 0.36, -0.02, -0.15,  0.26,  0.25, -0.16,  0.17
        },
        { 0.13,  0.5 ,  0.45, -0.07, -0.12, -0.37, -0.06, -0.07,
          0.07, -0.25, -0.28, -0.29, -0.3 ,  0.11,  0.13,  0.24},
        { 0.34,  0.03,  0.34,  0.09,  0.71,  0.18,  0.17, -0.24,
          -0.2 , 0.24,  0.1 ,  0.38,  0.15,  0.19,  0.09,  0.62}
    };
    int dims[] = {5, 16};
    tensor *image = tensor_init_from_data((float *)image_data, 2, dims);
    tensor *output = tensor_init(2, dims);
    tensor *ref = tensor_init_from_data((float *)ref_data, 2, dims);

    tensor_dct2d_rect(image, output, 0, 0, 5, 16);
    tensor_check_equal(output, ref, 0.01);

    tensor_free(image);
    tensor_free(ref);
    tensor_free(output);
}

void
test_dct8() {
    float x[8] = {20, 9, 10, 11, 12, 13, 14, 15};
    float y[8], y2[8];
    float y_exp[8] = {
        36.77, -0.56, 5.54, 4.32,
        4.24, 3.13, 2.30, 1.12
    };
    tensor_dct8_nvidia(x, y);
    tensor_dct8_loeffler(x, y2);
    for (int i = 0; i < 8; i++) {
        assert(approx_eq2(y_exp[i], y[i], 0.01));
        assert(approx_eq2(y_exp[i], y2[i], 0.01));
    }

    for (int i = 0; i < 8; i++) {
        x[i] = -10 + rand_n(20);
    }
    tensor_dct8_nvidia(x, y);
    tensor_dct8_loeffler(x, y2);
    for (int i = 0; i < 8; i++) {
        assert(approx_eq2(y[i], y2[i], 0.01));
    }
}

void
test_8x8_loeffler() {
    int dims[] = {8,  8};
    tensor *image = tensor_init_from_data((float *)IMAGE_DATA8X8, 2, dims);
    tensor *output = tensor_init(2, dims);
    tensor *output2 = tensor_init(2, dims);

    float y[8], y_exp[8] = {
        200.5, -0.4, 42.4, -2.7,
        -0.4, -30.8, -14.9, 54.8
    };
    tensor_dct8_nvidia(image->data, y);
    for (int i = 0; i < 8; i++) {
        assert(approx_eq2(y[i], y_exp[i], 0.1));
    }
    tensor_dct2d_8x8_blocks_loeffler(image, output);
    tensor_dct2d_blocks(image, output2, 8, 8);
    assert(tensor_check_equal(output, output2, 0.1));

    tensor_free(image);
    tensor_free(output);
    tensor_free(output2);

    dims[0] = 32;
    dims[1] = 32;
    image = tensor_init_from_data((float *)IMAGE_DATA32X32, 2, dims);
    output = tensor_init(2, dims);
    output2 = tensor_init(2, dims);

    tensor_dct2d_8x8_blocks_loeffler(image, output);
    tensor_dct2d_blocks(image, output2, 8, 8);
    assert(tensor_check_equal(output, output2, 0.1));

    tensor_free(image);
    tensor_free(output);
    tensor_free(output2);
}

int
main(int argc, char *argv[]) {
    rand_init(1234);
    PRINT_RUN(test_dct);
    PRINT_RUN(test_dct2);
    PRINT_RUN(test_dct_nonsquare);
    PRINT_RUN(test_dct8);
    PRINT_RUN(test_8x8_loeffler);
}
