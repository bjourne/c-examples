// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xmmintrin.h>
#include "datatypes/common.h"
#include "multiply.h"

// Should put this somewhere.
static int
ceil_div(int a, int b) {
    return a / b + (a % b != 0);
}

void
tensor_multiply_ref(tensor *a, tensor *b, tensor *c) {
    assert(a->n_dims == b->n_dims);
    assert(a->n_dims == 2);

    int a_rows = a->dims[0];
    int a_cols = a->dims[1];
    int b_rows = b->dims[0];
    int b_cols = b->dims[1];

    assert(a_cols == b_rows);
    assert(tensor_n_elements(c) == a_rows * b_cols);

    float *a_buf = a->data;
    float *b_buf = b->data;
    float *c_buf = c->data;

    memset(c_buf, 0, sizeof(float) * tensor_n_elements(c));

    for (int i = 0; i < a_rows; i++) {
        for (int k = 0; k < b_rows; k++) {
            for (int j = 0; j < b_cols; j++) {
                c_buf[b_cols * i + j] +=
                    a_buf[a_cols * i + k] * b_buf[k * b_cols + j];
            }
        }
    }
}

#define TILE_I          32
#define TILE_J          32
#define TILE_K          32
#define SIMD_HEIGHT     2
#define SIMD_WIDTH      16

static void
mul_fast_tile_16x2(
    int i0, int i1,
    int j0, int j1,
    int k0, int k1,
    float * restrict Apt,
    float * restrict Bpt,
    float * restrict C,
    int a_cols,
    int b_padded_cols
) {
    for (int i = i0; i < i1; i += SIMD_HEIGHT) {
        float * restrict b_ptr = Bpt;
        float * restrict Cptr0 = &C[b_padded_cols * (i + 0) + j0];
        float * restrict Cptr1 = &C[b_padded_cols * (i + 1) + j0];
        for (int j = j0; j < j1; j += SIMD_WIDTH) {
            __m128 acc00 = _mm_load_ps(Cptr0 + 0);
            __m128 acc01 = _mm_load_ps(Cptr0 + 4);
            __m128 acc02 = _mm_load_ps(Cptr0 + 8);
            __m128 acc03 = _mm_load_ps(Cptr0 + 12);

            __m128 acc10 = _mm_load_ps(Cptr1 + 0);
            __m128 acc11 = _mm_load_ps(Cptr1 + 4);
            __m128 acc12 = _mm_load_ps(Cptr1 + 8);
            __m128 acc13 = _mm_load_ps(Cptr1 + 12);

            float * restrict a_ptr = &Apt[a_cols * i + SIMD_HEIGHT * k0];
            for (int k = k0; k < k1; k++) {
                __m128 a0 = _mm_set1_ps(*a_ptr++);
                __m128 a1 = _mm_set1_ps(*a_ptr++);
                __m128 b0 = _mm_load_ps(b_ptr + 0);
                __m128 b1 = _mm_load_ps(b_ptr + 4);
                __m128 b2 = _mm_load_ps(b_ptr + 8);
                __m128 b3 = _mm_load_ps(b_ptr + 12);
                b_ptr += SIMD_WIDTH;
                acc00 = _mm_add_ps(acc00, _mm_mul_ps(a0,  b0));
                acc01 = _mm_add_ps(acc01, _mm_mul_ps(a0,  b1));
                acc02 = _mm_add_ps(acc02, _mm_mul_ps(a0,  b2));
                acc03 = _mm_add_ps(acc03, _mm_mul_ps(a0,  b3));

                acc10 = _mm_add_ps(acc10, _mm_mul_ps(a1,  b0));
                acc11 = _mm_add_ps(acc11, _mm_mul_ps(a1,  b1));
                acc12 = _mm_add_ps(acc12, _mm_mul_ps(a1,  b2));
                acc13 = _mm_add_ps(acc13, _mm_mul_ps(a1,  b3));
            }
            _mm_store_ps(Cptr0 +  0, acc00);
            _mm_store_ps(Cptr0 +  4, acc01);
            _mm_store_ps(Cptr0 +  8, acc02);
            _mm_store_ps(Cptr0 + 12, acc03);

            _mm_store_ps(Cptr1 +  0, acc10);
            _mm_store_ps(Cptr1 +  4, acc11);
            _mm_store_ps(Cptr1 +  8, acc12);
            _mm_store_ps(Cptr1 + 12, acc13);

            Cptr0 += SIMD_WIDTH;
            Cptr1 += SIMD_WIDTH;
        }
    }
}

typedef struct {
    float *a_tiled;
    float *b_tiled;
    float *c_padded;

    int start_i, end_i;
    // Height and width of padded c.
    int padded_height, padded_width;

    // Temp
    int a_cols;
} mul_job_t;

static void *
mul_thread(void *arg) {
    mul_job_t *job = (mul_job_t *)arg;

    int a_cols = job->a_cols;
    int b_padded_cols = job->padded_width;

    assert(job->start_i % TILE_I == 0);
    assert(job->end_i % TILE_I == 0);

    for (int i = job->start_i; i < job->end_i; i += TILE_I) {
        for  (int j = 0; j < job->padded_width; j += TILE_J) {
            for (int k = 0; k < job->padded_height; k += TILE_K) {
                float *Bptr = &job->b_tiled[k * b_padded_cols + j * TILE_K];
                mul_fast_tile_16x2(i, i + TILE_I,
                                   j, j + TILE_J,
                                   k, k + TILE_K,
                                   job->a_tiled, Bptr, job->c_padded,
                                   a_cols, b_padded_cols);
            }
        }
    }
    return 0;
}

void
tensor_multiply(tensor *a, tensor *b, tensor *c) {
    assert(a->n_dims == b->n_dims);
    assert(a->n_dims == 2);

    int a_rows = a->dims[0];
    int a_cols = a->dims[1];
    int b_rows = b->dims[0];
    int b_cols = b->dims[1];
    int c_rows = c->dims[0];
    int c_cols = c->dims[1];

    assert(a_cols == b_rows);
    assert(c_rows == a_rows);
    assert(c_cols == b_cols);

    tensor *a_tiled = tensor_transpose_a_new(a, SIMD_HEIGHT);
    tensor *b_tiled = tensor_transpose_b_new(b, TILE_K, SIMD_WIDTH);
    float *a_tiled_data = a_tiled->data;
    float *b_tiled_data = b_tiled->data;
    float *c_buf = calloc(a_rows * b_cols * sizeof(float), 1);

    long n_threads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * n_threads);
    mul_job_t *jobs = (mul_job_t *)malloc(sizeof(mul_job_t) * n_threads);

    int n_y_tiles = ceil_div(a_rows, TILE_I);
    int n_y_tiles_per_thread = ceil_div(n_y_tiles, n_threads);

    printf("%d a_rows, %d y tiles, %d y tiles/thread, %ld threads\n",
           a_rows,
           n_y_tiles,
           n_y_tiles_per_thread, n_threads);

    for (int i = 0; i < n_threads; i++) {
        int start_i = TILE_I * i * n_y_tiles_per_thread;
        int end_i = MIN(start_i + TILE_I * n_y_tiles_per_thread, a_rows);

        printf("%d -> %d\n", start_i, end_i);

        jobs[i] = (mul_job_t){
            a_tiled_data, b_tiled_data, c_buf,
            start_i, end_i,
            a_rows, b_cols, a_cols
        };
        pthread_create(&threads[i], NULL, mul_thread, &jobs[i]);
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int y = 0; y < c->dims[0]; y++) {
        for (int x = 0; x < c->dims[1]; x++) {
            c->data[y * c_cols + x] = c_buf[y * c_cols + x];
        }
    }

    free(threads);
    free(jobs);
    free(c_buf);

    tensor_free(a_tiled);
    tensor_free(b_tiled);
}

// Ofc this function can be generalized to higher dimensions, but I
// only need to linearize 2d tiles.
void
tensor_linearize_tiles(tensor *src, tensor *dst,
                       int tile_height, int tile_width) {
    assert(src->n_dims == 2 &&  src->n_dims == dst->n_dims);
    assert(tensor_n_elements(src) == tensor_n_elements(dst));
    int src_height = src->dims[0];
    int src_width = src->dims[1];
    assert(src_height % tile_height == 0);
    assert(src_width % tile_width == 0);

    float *src_buf = src->data;
    float *dst_buf = dst->data;

    for (int i = 0; i < src_height; i += tile_height) {
        for (int j = 0; j < src_width; j += tile_width) {
            for (int k = 0; k < tile_height; k++) {
                for (int l = 0; l < tile_width; l++) {
                    *dst_buf++ = src_buf[(i + k) * src_width + (j + l)];
                }
            }
        }
    }
}

tensor *
tensor_transpose_a_new(tensor *src, int simd_height) {
    int src_height = src->dims[0];
    int src_width = src->dims[1];

    int dst_height = ceil_div(src_height, simd_height) * src_width;
    tensor *dst = tensor_init(2, (int[]){dst_height, simd_height});

    float *src_ptr = src->data;
    float *dst_ptr = dst->data;
    for (int y = 0; y < src_height; y += simd_height) {
        for (int x = 0; x < src_width; x++) {
            for (int z = 0; z < simd_height; z++) {
                int src_y = y + z;
                if (src_y < src_height) {
                    *dst_ptr++ = src_ptr[src_width * src_y + x];
                } else {
                    *dst_ptr++ = 0.0;
                }
            }
        }
    }
    return dst;
}

// Combine with tensor_linearize_tiles?
tensor *
tensor_transpose_b_new(tensor *src, int tile_height, int tile_width) {
    int src_height = src->dims[0];
    int src_width = src->dims[1];

    int tiles_x = ceil_div(src_width, tile_width);
    int tiles_y = ceil_div(src_height, tile_height);

    int dst_height = tiles_x * tiles_y;
    int dst_width = tile_height * tile_width;
    tensor *dst = tensor_init(2, (int[]){dst_height, dst_width});

    float *src_ptr = src->data;
    float *dst_ptr = dst->data;
    for (int k = 0; k < src_height; k += tile_height) {
        for  (int j = 0; j < src_width; j += tile_width) {
            for (int x = 0; x < tile_height; x++ ) {
                for (int z = 0; z < tile_width; z++) {
                    int src_x = j + z;
                    int src_y = k + x;
                    if (src_x < src_width && src_y < src_height) {
                        *dst_ptr++ = src_ptr[src_width * src_y + src_x];
                    } else {
                        *dst_ptr++ = 0.0;
                    }
                }
            }
        }
    }
    return dst;
}
