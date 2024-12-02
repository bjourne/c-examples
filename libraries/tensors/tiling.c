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

// Unsigned types is better for clang (does it really matter???).
tensor *
tensor_tile_2d_mt_new(
    tensor *src,
    unsigned int tile_y,
    unsigned int tile_x,
    unsigned int fill_y,
    unsigned int fill_x
) {
    unsigned int src_y = src->dims[0];
    unsigned int src_x = src->dims[1];

    fill_y = fill_y ? fill_y : src_y;
    fill_x = fill_x ? fill_x : src_x;

    unsigned int n_tiles_y = CEIL_DIV(fill_y, tile_y);
    unsigned int n_tiles_x = CEIL_DIV(fill_x, tile_x);
    tensor *dst = tensor_init_4d(n_tiles_y, n_tiles_x, tile_y, tile_x);
    float *src_data = src->data;
    float *dst_data = dst->data;

    long n_jobs = 3 * sysconf(_SC_NPROCESSORS_ONLN);
    tile_job_t *jobs = malloc(sizeof(tile_job_t) * n_jobs);
    float y_tiles_per_thread = (float)n_tiles_y / (float)n_jobs;

    unsigned int start_i = 0;
    for (int i = 0; i < n_jobs; i++) {
        unsigned int end_i = ceil(y_tiles_per_thread * (i + 1)) * tile_y;
        jobs[i] = (tile_job_t){
            0, start_i, end_i,
            src_y, src_x,
            tile_y, tile_x,
            fill_y, fill_x,
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

void
tensor_tile_2d(tensor *src, tensor *dst) {
    assert(src->n_dims == 2 && dst->n_dims == 4);
    int n_tiles_y = dst->dims[0];
    int n_tiles_x = dst->dims[1];
    int tile_y = dst->dims[2];
    int tile_x = dst->dims[3];

    int src_y = src->dims[0];
    int src_x = src->dims[1];

    float *s_ptr = src->data;
    float *d_ptr = dst->data;
    for (int k = 0; k < n_tiles_y; k++) {
        for  (int j = 0; j < n_tiles_x; j++) {
            for (int x = 0; x < tile_y; x++ ) {
                for (int z = 0; z < tile_x; z++) {
                    int at_x = tile_x*j + z;
                    int at_y = tile_y*k + x;
                    float v = 0.0;
                    if (at_x < src_x && at_y < src_y) {
                        v = s_ptr[src_x * at_y + at_x];
                    }
                    *d_ptr++ = v;
                }
            }
        }
    }
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
