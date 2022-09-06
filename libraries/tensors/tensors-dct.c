// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensors-dct.h"

static float SQRT2 = 1.4142135623730951f;
static float SQRT2INV = 0.7071067811865475f;

extern inline void tensor_dct8_loeffler(float x[8], float y[8]);
extern inline void tensor_dct8_nvidia(float * restrict x,
                                      float * restrict y);

void
tensor_dct2d_rect(tensor *src, tensor *dst,
                  int sy, int sx,
                  int height, int width) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int n_cols = src->dims[1];
    float pi_div_2rows = 0.5 * M_PI / height;
    float pi_div_2cols = 0.5 * M_PI / width;
    float coeff = 2.0 / sqrt(height * width);

    for (int u = 0; u < height; u++) {
        for (int v = 0; v < width; v++) {
            float o = 0.0f;
            for (int y = 0; y < height;  y++) {
                for (int x = 0; x < width; x++) {
                    float cos_y = cos((2 * y + 1) * u * pi_div_2rows);
                    float cos_x = cos((2 * x + 1) * v * pi_div_2cols);
                    int r_addr = n_cols * (y + sy) + x + sx;
                    o  += src->data[r_addr] * cos_y * cos_x;
                }
            }
            float c_u = u == 0 ? SQRT2INV : 1;
            float c_v = v == 0 ? SQRT2INV : 1;
            int w_addr = n_cols * (u + sy) + v + sx;
            dst->data[w_addr] = coeff * c_u * c_v * o;
        }
    }
}

void
tensor_dct2d_blocks(tensor *src, tensor *dst,
                    int block_height, int block_width) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int height = src->dims[0];
    int width = src->dims[1];

    // Lazyness
    assert(width % block_width == 0);
    assert(height % block_height == 0);

    for (int y = 0; y < height; y += block_height) {
        for (int x = 0; x < width; x += block_width) {
            tensor_dct2d_rect(src, dst, y, x, block_height, block_width);
        }
    }
}

void
tensor_dct2d_8x8_blocks_nvidia_avx256(tensor *src, tensor *dst) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int height = src->dims[0];
    int width = src->dims[1];

    tensor_dct8x8_nvidia_avx256_impl(src->data, dst->data, width, height);
}

void
tensor_idct2d(tensor *src, tensor *dst) {
    assert(src->n_dims == dst->n_dims  && src->n_dims == 2);
    assert(src->dims[0] == dst->dims[0]);
    assert(src->dims[1] == dst->dims[1]);

    int n_rows = src->dims[0];
    int n_cols = src->dims[1];

    float pi_div_2rows = 0.5 * M_PI / n_rows;
    float pi_div_2cols = 0.5 * M_PI / n_cols;
    float denom = sqrt(n_rows * n_cols);
    for  (unsigned int u = 0; u < n_rows; u++) {
        for (unsigned int v = 0; v < n_cols; v++) {
            float o = 0.0f;
            for (unsigned int y = 0; y < n_rows; y++) {
                for (unsigned int x = 0; x < n_cols; x++) {
                    float a_y = y == 0 ? 1.0 : SQRT2;
                    float a_x = x == 0 ? 1.0 : SQRT2;
                    float cos_y = cos((2 * u + 1) * y * pi_div_2rows);
                    float cos_x = cos((2 * v + 1) * x * pi_div_2cols);
                    o += src->data[n_cols * y + x] * a_y * a_x * cos_y * cos_x / denom;
                }
            }
            dst->data[n_cols * u + v] = o;
        }
    }
}
