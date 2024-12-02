// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <immintrin.h>

#include "datatypes/common.h"
#include "multiply.h"
#include "tiling.h"

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
            float a = a_buf[a_cols * i + k];
            for (int j = 0; j < b_cols; j++) {
                c_buf[b_cols * i + j] += a * b_buf[b_cols * k + j];
            }
        }
    }
}

#if __AVX512F__

#define TILE_I          128
#define TILE_J          256
#define TILE_K          256

#define SIMD_HEIGHT     4
#define SIMD_WIDTH      64

#define N_FLOATS        16
#define SIMD_COLS       (SIMD_WIDTH / N_FLOATS)

#define UNROLL_FACTOR   32
#define JOB_FACTOR      2

static void
mul_fast_kernel(
    float * restrict a_base,
    float * restrict b_base,
    float * restrict c_base,
    unsigned int K, unsigned int M
) {
    for (unsigned int i = 0; i < TILE_I / SIMD_HEIGHT; i++) {
        float * restrict b_ptr = b_base;
        float * restrict c_ptr[SIMD_HEIGHT];
        for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
            c_ptr[y] = c_base + M * y;
        }
        for (unsigned int j = 0; j < TILE_J / SIMD_WIDTH; j++) {
            __m512 acc[SIMD_HEIGHT][SIMD_COLS];
            for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                for (unsigned int x = 0; x < SIMD_COLS; x++) {
                    acc[y][x] = _mm512_load_ps(c_ptr[y] + N_FLOATS * x);
                }
            }
            float * restrict a_ptr = a_base;
            #ifdef __clang__
            #pragma unroll UNROLL_FACTOR
            #else
            #pragma GCC unroll 32
            #endif
            for (unsigned int k = 0; k < TILE_K; k++) {
                __m512 b[SIMD_COLS];
                for (unsigned int x = 0; x < SIMD_COLS; x++) {
                    b[x] = _mm512_load_ps(b_ptr + N_FLOATS * x);
                }
                b_ptr += SIMD_WIDTH;
                __m512 a[SIMD_HEIGHT];
                for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                    a[y] = _mm512_set1_ps(*(a_ptr + y));
                }
                a_ptr += SIMD_HEIGHT;
                for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                    for (unsigned int x = 0; x < SIMD_COLS; x++) {
                        acc[y][x] = _mm512_fmadd_ps(a[y], b[x], acc[y][x]);
                    }
                }
            }
            for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                for (unsigned int x = 0; x < SIMD_COLS; x++) {
                    _mm512_store_ps(c_ptr[y] + N_FLOATS * x, acc[y][x]);
                }
            }
            for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                c_ptr[y] += SIMD_WIDTH;
            }
        }
        a_base += SIMD_HEIGHT * K;
        c_base += SIMD_HEIGHT * M;
    }
}

#else

#define TILE_I          256
#define TILE_J          256
#define TILE_K          256

#define SIMD_HEIGHT     2
#define SIMD_WIDTH      16
#define SIMD_COLS       (SIMD_WIDTH / 4)

#define JOB_FACTOR      3

static void
mul_fast_kernel(
    float * restrict a_base,
    float * restrict b_base,
    float * restrict c_base,
    unsigned int K, unsigned int M
) {
    for (unsigned int i = 0; i < TILE_I; i += SIMD_HEIGHT) {
        float * restrict b_ptr = b_base;
        float * restrict c_ptr[SIMD_HEIGHT];
        for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
            c_ptr[y] = c_base + M * y;
        }
        for (unsigned int j = 0; j < TILE_J; j += SIMD_WIDTH) {
            __m128 acc[SIMD_HEIGHT][SIMD_COLS];
            for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                for (unsigned int x = 0; x < SIMD_COLS; x++) {
                    acc[y][x] = _mm_load_ps(c_ptr[y] + 4 * x);
                }
            }
            float * restrict a_ptr = a_base;

            for (unsigned int k = 0; k < TILE_K; k++) {
                __m128 b[SIMD_COLS];
                for (unsigned int x = 0; x < SIMD_COLS; x++) {
                    b[x] = _mm_load_ps(b_ptr + 4 * x);
                }
                b_ptr += SIMD_WIDTH;
                __m128 a[SIMD_HEIGHT];
                for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                    a[y] = _mm_set1_ps(*(a_ptr + y));
                }
                a_ptr += SIMD_HEIGHT;
                for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                    for (unsigned int x = 0; x < SIMD_COLS; x++) {
#if defined(__AVX2__)
                        acc[y][x] = _mm_fmadd_ps(a[y], b[x], acc[y][x]);
#else

                        acc[y][x] = _mm_add_ps(acc[y][x],
                                               _mm_mul_ps(a[y], b[x]));

#endif
                    }
                }
            }
            for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                for (unsigned int x = 0; x < SIMD_COLS; x++) {
                    _mm_store_ps(c_ptr[y] + 4 * x, acc[y][x]);
                }
            }
            for (unsigned int y = 0; y < SIMD_HEIGHT; y++) {
                c_ptr[y] += SIMD_WIDTH;
            }
        }
        a_base += SIMD_HEIGHT * K;
        c_base += SIMD_HEIGHT * M;
    }
}

#endif

typedef struct {
    pthread_t thread;
    float *a_tiled;
    float *b_tiled;
    float *c_buf;

    unsigned int start_i, end_i;
    unsigned int M, K;
} mul_job;

static void
mul_tiles(float * restrict a_tiled,
          float * restrict b_tiled,
          float * restrict c_buf,
          unsigned int start_i, unsigned int end_i,
          unsigned int M, unsigned int K) {
    for (unsigned int i = start_i; i < end_i; i += TILE_I) {
        for (unsigned int j = 0; j < M; j += TILE_J) {
            float *c_base = &c_buf[M * i + j];
            for (unsigned int k = 0; k < K; k += TILE_K) {
                float *a_base = &a_tiled[K * i + k * SIMD_HEIGHT];
                float *b_base = &b_tiled[M * k + j * TILE_K];
                mul_fast_kernel(
                    a_base, b_base, c_base,
                    K, M);
            }
        }
    }
}

static void *
mul_thread(void *arg) {
    mul_job job = *(mul_job *)arg;
    mul_tiles(job.a_tiled, job.b_tiled, job.c_buf,
              job.start_i, job.end_i,
              job.M, job.K);
    return NULL;
}

void
tensor_multiply_w_params(tensor *a, tensor *b, tensor *c, int n_jobs) {
    int a_rows = a->dims[0];
    int a_cols = a->dims[1];
    int b_rows = b->dims[0];
    int b_cols = b->dims[1];
    int c_rows = c->dims[0];
    int c_cols = c->dims[1];

    assert(a->n_dims == b->n_dims);
    assert(a->n_dims == 2);
    assert(a_cols == b_rows);
    assert(a_rows == c_rows);
    assert(b_cols == c_cols);
    assert(TILE_I % SIMD_HEIGHT == 0);
    assert(TILE_J % SIMD_WIDTH == 0);

    int n_i_tiles = CEIL_DIV(a_rows, TILE_I);
    int N = n_i_tiles * TILE_I;
    int K = CEIL_DIV(a_cols, TILE_K) * TILE_K;
    int M = CEIL_DIV(b_cols, TILE_J) * TILE_J;

    tensor *a_tiled = tensor_tile_2d_mt_new(a, SIMD_HEIGHT, 1, N, K);
    tensor *b_tiled = tensor_tile_2d_mt_new(b, TILE_K, SIMD_WIDTH, K, M);

    float *a_tiled_data = a_tiled->data;
    float *b_tiled_data = b_tiled->data;

    float *c_buf = c->data;
    size_t n_c_bytes = N * M * sizeof(float);
    if (c_rows != N || c_cols != M) {
        c_buf = malloc_aligned(TENSOR_ADDRESS_ALIGNMENT, n_c_bytes);
    }
    memset(c_buf, 0, n_c_bytes);

    // Running more jobs than there are tiles leads to badness.
    n_jobs = MIN(n_jobs, n_i_tiles);
    if (n_jobs == 1) {
        mul_tiles(a_tiled_data, b_tiled_data, c_buf,
                  0, N, M, K);
    } else {
        mul_job *jobs = malloc(sizeof(mul_job) * n_jobs);
        float i_tiles_per_thread = (float)n_i_tiles / (float)n_jobs;

        int start_i = 0;
        for (int i = 0; i < n_jobs; i++) {
            int end_i = ceil(i_tiles_per_thread * (i + 1)) * TILE_I;
            jobs[i] = (mul_job){0,
                                a_tiled_data, b_tiled_data, c_buf,
                                start_i, end_i,
                                M, K
            };
            pthread_create(&jobs[i].thread, NULL, mul_thread, &jobs[i]);
            start_i = end_i;
        }
        for (int i = 0; i < n_jobs; i++) {
            pthread_join(jobs[i].thread, NULL);
        }
        free(jobs);
    }
    if (c_buf != c->data) {
        for (int y = 0; y < c_rows; y++) {
            memcpy(&c->data[y * c_cols],
                   &c_buf[y * M], c_cols * sizeof(float));
        }
        free(c_buf);
    }
    tensor_free(a_tiled);
    tensor_free(b_tiled);
}

void
tensor_multiply(tensor *a, tensor *b, tensor *c) {
    int n_jobs = JOB_FACTOR * (int)sysconf(_SC_NPROCESSORS_ONLN);
    tensor_multiply_w_params(a, b, c, n_jobs);
}
