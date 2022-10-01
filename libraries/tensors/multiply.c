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
static unsigned int
ceil_div(unsigned int a, unsigned int b) {
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

#define TILE_I          256
#define TILE_J          32
#define TILE_K          32
#define SIMD_HEIGHT     2
#define SIMD_WIDTH      16

static void
mul_fast_tile_16x2(
    float * restrict a_base,
    float * restrict b_base,
    float * restrict c_base,
    int K, int M
) {
    for (int i = 0; i < TILE_I; i += SIMD_HEIGHT) {
        float * restrict b_ptr = b_base;
        float * restrict c_ptr0 = c_base;
        float * restrict c_ptr1 = c_base + M;
        for (int j = 0; j < TILE_J; j += SIMD_WIDTH) {
            __m128 acc00 = _mm_load_ps(c_ptr0 + 0);
            __m128 acc01 = _mm_load_ps(c_ptr0 + 4);
            __m128 acc02 = _mm_load_ps(c_ptr0 + 8);
            __m128 acc03 = _mm_load_ps(c_ptr0 + 12);

            __m128 acc10 = _mm_load_ps(c_ptr1 + 0);
            __m128 acc11 = _mm_load_ps(c_ptr1 + 4);
            __m128 acc12 = _mm_load_ps(c_ptr1 + 8);
            __m128 acc13 = _mm_load_ps(c_ptr1 + 12);

            float * restrict a_ptr = a_base;
            for (int k = 0; k < TILE_K; k++) {
                __m128 b0 = _mm_load_ps(b_ptr + 0);
                __m128 b1 = _mm_load_ps(b_ptr + 4);
                __m128 b2 = _mm_load_ps(b_ptr + 8);
                __m128 b3 = _mm_load_ps(b_ptr + 12);
                __m128 a0 = _mm_set1_ps(*a_ptr++);
                __m128 a1 = _mm_set1_ps(*a_ptr++);
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
            _mm_store_ps(c_ptr0 +  0, acc00);
            _mm_store_ps(c_ptr0 +  4, acc01);
            _mm_store_ps(c_ptr0 +  8, acc02);
            _mm_store_ps(c_ptr0 + 12, acc03);

            _mm_store_ps(c_ptr1 +  0, acc10);
            _mm_store_ps(c_ptr1 +  4, acc11);
            _mm_store_ps(c_ptr1 +  8, acc12);
            _mm_store_ps(c_ptr1 + 12, acc13);

            c_ptr0 += SIMD_WIDTH;
            c_ptr1 += SIMD_WIDTH;
        }
        a_base += SIMD_HEIGHT * K;
        c_base += SIMD_HEIGHT * M;
    }
}

typedef struct {
    pthread_t thread;
    float *a_tiled;
    float *b_tiled;
    float *c_buf;

    int start_i, end_i;
    int M, K;
} mul_job_t;

static void *
mul_thread(void *arg) {
    mul_job_t job = *(mul_job_t *)arg;
    int K = job.K;
    int M = job.M;
    int start_i = job.start_i;
    int end_i = job.end_i;
    float *a_tiled = job.a_tiled;
    float *b_tiled = job.b_tiled;
    float *c_buf = job.c_buf;

    //printf("%d -> %d\n", start_i, end_i);
    for (int i = start_i; i < end_i; i += TILE_I) {
        for (int j = 0; j < M; j += TILE_J) {
            for (int k = 0; k < K; k += TILE_K) {
                float *a_base = &a_tiled[K * i + k * SIMD_HEIGHT];
                float *b_base = &b_tiled[M * k + j * TILE_K];
                float *c_base = &c_buf[M * i + j];
                mul_fast_tile_16x2(
                    a_base, b_base, c_base,
                    K, M);
            }
        }
    }
    return NULL;
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
    assert(a_rows == c_rows);
    assert(b_cols == c_cols);

    // The K dimension doesn't need to be divisble by TILE_K.
    int N = ceil_div(a_rows, TILE_I) * TILE_I;
    int K = a_cols;
    int M = ceil_div(b_cols, TILE_J) * TILE_J;

    // I think A needs to be tiled to N rows.
    tensor *a_tiled = tensor_linearize_tiles_new2(a, SIMD_HEIGHT, 1,
                                                  N, a_cols);
    tensor *b_tiled = tensor_linearize_tiles_new2(b, TILE_K, SIMD_WIDTH,
                                                  K, M);

    float *a_tiled_data = a_tiled->data;
    float *b_tiled_data = b_tiled->data;
    float *c_buf = calloc(N * M, sizeof(float));
    long n_jobs = sysconf(_SC_NPROCESSORS_ONLN);
    mul_job_t *jobs = (mul_job_t *)malloc(sizeof(mul_job_t) * n_jobs);

    int n_y_tiles = ceil_div(N, TILE_I);
    float y_tiles_per_thread = (float)n_y_tiles / (float)n_jobs;

    int y_start = 0;
    for (int i = 0; i < n_jobs; i++) {
        int y_end = ceil(y_tiles_per_thread * (i + 1)) * TILE_I;
        jobs[i] = (mul_job_t){0,
            a_tiled_data, b_tiled_data, c_buf,
            y_start, y_end,
            M, K
        };
        pthread_create(&jobs[i].thread, NULL, mul_thread, &jobs[i]);
        y_start = y_end;
    }
    for (int i = 0; i < n_jobs; i++) {
        pthread_join(jobs[i].thread, NULL);
    }
    for (int y = 0; y < c_rows; y++) {
        memcpy(&c->data[y * c_cols], &c_buf[y * M], c_cols * sizeof(float));
    }
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

typedef struct {
    pthread_t thread;
    unsigned int start_i, end_i;

    unsigned int src_height, src_width;
    unsigned int tile_height, tile_width;
    unsigned int fill_height, fill_width;

    float *src;
    float *dst;
} tile_job_t;

static void *
tile_thread(void *arg) {
    tile_job_t job = *(tile_job_t *)arg;

    unsigned int src_height = job.src_height;
    unsigned int src_width = job.src_width;
    unsigned int tile_height = job.tile_height;
    unsigned int tile_width = job.tile_width;
    unsigned int fill_width = job.fill_width;
    unsigned int start_i = job.start_i;
    unsigned int end_i = job.end_i;
    float *src = job.src;
    float *dst = job.dst;

    for (unsigned int i = start_i; i < end_i; i += tile_height) {
        float *dst_ptr = &dst[fill_width * i];
        for (unsigned int j = 0; j < fill_width; j += tile_width) {
            for (unsigned int y = i; y < i + tile_height; y++ ) {
                for (unsigned int x = j; x < j + tile_width; x++) {
                    if (x < src_width && y < src_height) {
                        *dst_ptr++ = src[src_width * y + x];
                    } else {
                        *dst_ptr++ = 0.0;
                    }
                }
            }
        }
    }
    return NULL;
}

// Unsigned types is better for clang.
tensor *
tensor_linearize_tiles_new2(
    tensor *src,
    unsigned int tile_height,
    unsigned int tile_width,
    unsigned int fill_height,
    unsigned int fill_width
) {

    unsigned int src_height = src->dims[0];
    unsigned int src_width = src->dims[1];

    unsigned int tiles_x = ceil_div(fill_width, tile_width);
    unsigned int tiles_y = ceil_div(fill_height, tile_height);

    unsigned int dst_height = tiles_x * tiles_y;
    unsigned int dst_width = tile_height * tile_width;
    tensor *dst = tensor_init(2, (int[]){dst_height, dst_width});

    float *src_data = src->data;
    float *dst_data = dst->data;

    long n_jobs = sysconf(_SC_NPROCESSORS_ONLN);
    tile_job_t *jobs = malloc(sizeof(tile_job_t) * n_jobs);

    unsigned int n_y_tiles = ceil_div(fill_height, tile_height);
    float y_tiles_per_thread = (float)n_y_tiles / (float)n_jobs;

    unsigned int start_i = 0;
    for (int i = 0; i < n_jobs; i++) {
        unsigned int end_i = ceil(y_tiles_per_thread * (i + 1)) * tile_height;
        jobs[i] = (tile_job_t){
            0, start_i, end_i,
            src_height, src_width,
            tile_height, tile_width,
            fill_height, fill_width,
            src_data, dst_data
        };
        pthread_create(&jobs[i].thread, NULL, tile_thread, &jobs[i]);
        start_i = end_i;
    }
    for (int i = 0; i < n_jobs; i++) {
        pthread_join(jobs[i].thread, NULL);
    }
    free(jobs);
    return dst;
}

tensor *
tensor_linearize_tiles_new(tensor *src,
                           int tile_height, int tile_width) {
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
