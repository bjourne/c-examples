#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "datatypes/common.h"
#include "multiply.h"

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

#define TILE_I  32

typedef struct {
    tensor *a, *b, *c;
    int start_i, end_i;
} mul_job_t;

static void *
mul_thread(void *arg) {
    mul_job_t *job = (mul_job_t *)arg;

    tensor *a = job->a;
    tensor *b = job->b;
    tensor *c = job->c;

    int a_cols = a->dims[1];
    int b_rows = b->dims[0];
    int b_cols = b->dims[1];

    float *a_buf = a->data;
    float *b_buf = b->data;
    float *c_buf = c->data;

    for (int i = job->start_i; i < job->end_i; i++) {
        for (int k = 0; k < b_rows; k++) {
            for (int j = 0; j < b_cols; j++) {
                c_buf[b_cols * i + j] +=
                    a_buf[a_cols * i + k] * b_buf[k * b_cols + j];
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
    assert(tensor_n_elements(c) == a_rows * b_cols);

    float *c_buf = c->data;
    memset(c_buf, 0, sizeof(float) * tensor_n_elements(c));

    long n_threads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * n_threads);
    mul_job_t *jobs = (mul_job_t *)malloc(sizeof(mul_job_t) * n_threads);

    int n_tile_rows = (int)ceil((float)c_rows / (float)TILE_I);
    int n_tiles_per_thread =
        (int)ceil((float)n_tile_rows / (float)n_threads);

    printf("%ld threads, %d rows, %d tile rows, %d tiles/thread\n",
           n_threads, c_rows, n_tile_rows, n_tiles_per_thread);
    for (int i = 0; i < n_threads; i++) {
        int start_i = TILE_I * i * n_tiles_per_thread;
        int end_i = MIN(start_i + TILE_I * n_tiles_per_thread, c_rows);
        jobs[i] = (mul_job_t){a, b, c,
                              start_i, end_i};
        pthread_create(&threads[i], NULL, mul_thread, &jobs[i]);
        printf("%d, %d\n", start_i, end_i);
    }
    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    free(jobs);
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
