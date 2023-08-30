// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Generates an image of the Mandelbrot set.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "linalg/linalg-simd.h"
#include "datatypes/common.h"
#include "tensors/tensors.h"

#define HEIGHT (3 * 1024)
#define WIDTH (3 * 1920)
#define N_PIXELS (HEIGHT * WIDTH)
#define MAX_ITER 500

static void
mandelbrot_avx2(float *ptr, float _y, float _min_x, float _scale_x) {
    assert(WIDTH % 8 == 0);
    float8 iota = f8_set_8x(0, 1, 2, 3, 4, 5, 6, 7);
    float8 min_x = f8_set_1x(_min_x);
    float8 scale_x = f8_set_1x(_scale_x);
    float8 y = f8_set_1x(_y);
    for (uint32_t _w = 0; _w < WIDTH; _w += 8) {
        float8 w = f8_add(iota, f8_set_1x(_w));
        float8 x = f8_add(min_x, f8_mul(w, scale_x));
        float8 u = f8_0();
        float8 v = f8_0();
        float8 u2 = f8_0();
        float8 v2 = f8_0();
        int8 cnt = i8_0();
        for (uint32_t i = 0; i < MAX_ITER; i++) {
            float8 mask = f8_cmp_lte(f8_add(u2, v2), f8_4());
            if (!f8_movemask(mask)) {
                break;
            }
            v = f8_fma(f8_add(u, u), v, y);
            u = f8_add(f8_sub(u2, v2), x);
            u2 = f8_mul(u, u);
            v2 = f8_mul(v, v);
            cnt = i8_sub(cnt, (int8)mask);
        }
        float8 fcnt = f8_set_8x_i8(i8_abs(cnt));
        fcnt = f8_div(fcnt, f8_set_1x(MAX_ITER));
        f8_store(fcnt, ptr);
        ptr += 8;
    }
}

static void
mandelbrot_sse(float *ptr, float _y, float _min_x, float _scale_x) {
    assert(WIDTH % 4 == 0);
    float4 iota = f4_set_4x(0, 1, 2, 3);
    float4 min_x = f4_set_1x(_min_x);
    float4 scale_x = f4_set_1x(_scale_x);
    float4 y = f4_set_1x(_y);
    for (uint32_t _w = 0; _w < WIDTH; _w += 4) {
        float4 w = f4_add(iota, f4_set_1x(_w));
        float4 x = f4_add(min_x, f4_mul(w, scale_x));
        float4 u = f4_0();
        float4 v = f4_0();
        float4 u2 = f4_0();
        float4 v2 = f4_0();
        float4 cnt = f4_0();
        for (uint32_t i = 0; i < MAX_ITER; i++) {
            float4 mask = f4_cmp_lte(f4_add(u2, v2), f4_4());
            if (!f4_movemask(mask)) {
                break;
            }
            v = f4_fma(f4_add(u, u), v, y);
            u = f4_add(f4_sub(u2, v2), x);
            u2 = f4_mul(u, u);
            v2 = f4_mul(v, v);
            cnt = f4_add(cnt, f4_and(mask, f4_1()));
        }
        cnt = f4_div(cnt, f4_set_1x(MAX_ITER));
        f4_store(cnt, ptr);
        ptr += 4;
    }
}

static void
mandelbrot_scalar(float *ptr, float y, float min_x, float scale_x) {
    for (uint32_t w = 0; w < WIDTH; w++) {
        float x = min_x + w * scale_x;
        float u = 0.0;
        float v = 0.0;
        float u2 = 0.0;
        float v2 = 0.0;
        uint32_t i = 0;
        while (i < MAX_ITER && u2 + v2 <= 4) {
            v = 2*u*v + y;
            u = u2 - v2 + x;
            u2 = u*u;
            v2 = v*v;
            i++;
        }
        *ptr = (float)i / MAX_ITER;
        ptr++;
    }
}

static void
run_mandelbrot(uint32_t tp) {
    float min_x = -2.0, max_x = 0.47;
    float min_y = -1.12, max_y = 1.12;
    float scale_x = (max_x - min_x) / WIDTH;
    float scale_y = (max_y - min_y) / HEIGHT;
    uint64_t start = nano_count();
    float *pixels = malloc_aligned(64, N_PIXELS * sizeof(float));
    for (uint32_t h = 0; h < HEIGHT; h++) {
        float cy = min_y + h * scale_y;
        float *addr = &pixels[WIDTH * h];
        if (tp == 0) {
            mandelbrot_scalar(addr, cy, min_x, scale_x);
        } else if (tp == 1) {
            mandelbrot_sse(addr, cy, min_x, scale_x);
        } else if (tp == 2) {
            mandelbrot_avx2(addr, cy, min_x, scale_x);
        }
    }
    uint64_t delta = nano_count() - start;
    double nanos_per_pixel = (double)delta / N_PIXELS;
    double secs = (double)delta / (1000 * 1000 * 1000);

#ifdef HAVE_PNG
    tensor *t = tensor_init(3, (int[]){3, HEIGHT, WIDTH});
    tensor_fill_const(t, 0);
    for (uint32_t h = 0; h < HEIGHT; h++) {
        for (uint32_t w = 0; w < WIDTH; w++) {
            uint8_t col = pixels[WIDTH * h + w] * 255;
            t->data[N_PIXELS * tp + WIDTH * h + w] = col;
        }
    }
    char *fname = NULL;
    if (tp == 0) {
        fname = "mandelbrot-scalar.png";
    } else if (tp == 1) {
        fname = "mandelbrot-sse.png";
    } else if (tp == 2) {
        fname = "mandelbrot-avx2.png";
    }
    assert(tensor_write_png(t, fname));
    tensor_free(t);
#endif
    free(pixels);
    if (tp == 0) {
        printf("== Scalar ==\n");
    } else if (tp == 1) {
        printf("== SSE ==\n");
    } else if (tp == 2) {
        printf("== AVX2 ==\n");
    }
    printf("%.2f seconds\n", secs);
    printf("%.0f nanos/pixels\n", nanos_per_pixel);
}

int
main(int argc, char *argv[]) {
    run_mandelbrot(0);
    run_mandelbrot(1);
    run_mandelbrot(2);
    return 0;
}
