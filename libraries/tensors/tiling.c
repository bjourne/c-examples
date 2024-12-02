// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include "datatypes/common.h"
#include "tiling.h"

typedef struct {
    pthread_t thread;
    int lo_y, hi_y;
    tensor *src;
    tensor *dst;
} tile_job_t;

/* static void */
/* tile_y_range_even(float *src, float *dst, */
/*                   int src_y, int src_x, */
/*                   int y_tile_lo, int y_tile_hi, */
/*                   int n_tiles_y, int n_tiles_x, */
/*                   int tile_y, int tile_x) { */
/*     float *d_ptr = &dst[y_tile_lo * n_tiles_x * tile_x * tile_y]; */
/*     for (int k = y_tile_lo; k < y_tile_hi; k++) { */
/*         for  (int j = 0; j < n_tiles_x; j++) { */
/*             for (int x = 0; x < tile_y; x++ ) { */
/*                 for (int z = 0; z < tile_x; z++) { */
/*                     int at_x = tile_x*j + z; */
/*                     int at_y = tile_y*k + x; */
/*                     *d_ptr++ = src[src_x * at_y + at_x]; */
/*                 } */
/*             } */
/*         } */
/*     } */
/* } */

static inline void
tile_y_range(float *src, float *dst,
             int src_y, int src_x,
             int y_tile_lo, int y_tile_hi,
             int n_tiles_y, int n_tiles_x,
             int tile_y, int tile_x) {
    float *d_ptr = &dst[y_tile_lo * n_tiles_x * tile_x * tile_y];
    for (int k = y_tile_lo; k < y_tile_hi; k++) {
        for  (int j = 0; j < n_tiles_x; j++) {
            for (int x = 0; x < tile_y; x++ ) {
                for (int z = 0; z < tile_x; z++) {
                    int at_x = tile_x*j + z;
                    int at_y = tile_y*k + x;
                    float v = 0.0;
                    if (at_x < src_x && at_y < src_y) {
                        v = src[src_x * at_y + at_x];
                    }
                    *d_ptr++ = v;
                }
            }
        }
    }
}


static void *
tile_thread(void *arg) {
    tile_job_t job = *(tile_job_t *)arg;
    tensor *dst = job.dst;
    tensor *src = job.src;

    int n_tiles_y = dst->dims[0];
    int n_tiles_x = dst->dims[1];
    int tile_y = dst->dims[2];
    int tile_x = dst->dims[3];

    int src_y = src->dims[0];
    int src_x = src->dims[1];

    int lo_y = job.lo_y;
    int hi_y = job.hi_y;

    tile_y_range(src->data, dst->data,
                 src_y, src_x,
                 lo_y, hi_y,
                 n_tiles_y, n_tiles_x,
                 tile_y, tile_x);

    return NULL;
}

tensor *
tensor_tile_2d_mt_new(
    tensor *src,
    int tile_y,
    int tile_x,
    int fill_y,
    int fill_x
) {
    int src_y = src->dims[0];
    int src_x = src->dims[1];

    fill_y = fill_y ? fill_y : src_y;
    fill_x = fill_x ? fill_x : src_x;

    int n_tiles_y = CEIL_DIV(fill_y, tile_y);
    int n_tiles_x = CEIL_DIV(fill_x, tile_x);
    tensor *dst = tensor_init_4d(n_tiles_y, n_tiles_x, tile_y, tile_x);

    long n_jobs = MIN(3 * sysconf(_SC_NPROCESSORS_ONLN), n_tiles_y);
    tile_job_t *jobs = malloc(sizeof(tile_job_t) * n_jobs);
    int n_y_tiles_per_thread = CEIL_DIV(n_tiles_y, n_jobs);

    for (int i = 0; i < n_jobs; i++) {
        int lo_y = i * n_y_tiles_per_thread;
        int hi_y = MIN(lo_y + n_y_tiles_per_thread, n_tiles_y);
        jobs[i] = (tile_job_t){
            0, lo_y, hi_y, src, dst
        };
        pthread_create(&jobs[i].thread, NULL, tile_thread, &jobs[i]);
    }
    for (int i = 0; i < n_jobs; i++) {
        pthread_join(jobs[i].thread, NULL);
    }
    free(jobs);
    return dst;
}

void
tensor_tile_2d(tensor *src, tensor *dst) {
    assert(src->n_dims == 2 && dst->n_dims == 4);
    int n_tiles_y = dst->dims[0];
    int n_tiles_x = dst->dims[1];
    int tile_y = dst->dims[2];
    int tile_x = dst->dims[3];

    int src_y = src->dims[0];
    int src_x = src->dims[1];

    tile_y_range(src->data, dst->data,
                 src_y, src_x,
                 0, n_tiles_y,
                 n_tiles_y, n_tiles_x,
                 tile_y, tile_x);
}

tensor *
tensor_tile_2d_new(
    tensor *src,
    int tile_y, int tile_x,
    int fill_y, int fill_x
) {
    assert(src->n_dims == 2);

    fill_y = fill_y ? fill_y : src->dims[0];
    fill_x = fill_x ? fill_x : src->dims[1];

    int n_tiles_y = CEIL_DIV(fill_y, tile_y);
    int n_tiles_x = CEIL_DIV(fill_x, tile_x);

    tensor *dst = tensor_init_4d(
        n_tiles_y, n_tiles_x, tile_y, tile_x
    );
    tensor_tile_2d(src, dst);
    return dst;
}
