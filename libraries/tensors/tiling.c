// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include "datatypes/common.h"
#include "tiling.h"

// Not all of these 2d tiling functions are necessary.
void
tensor_linearize_tiles(tensor *src, tensor *dst,
                       int tile_height, int tile_width) {
    assert(src->n_dims == 2 && src->n_dims == dst->n_dims);
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

    unsigned int tiles_x = CEIL_DIV(fill_width, tile_width);
    unsigned int tiles_y = CEIL_DIV(fill_height, tile_height);

    unsigned int dst_height = tiles_x * tiles_y;
    unsigned int dst_width = tile_height * tile_width;
    tensor *dst = tensor_init_2d(dst_height, dst_width);

    float *src_data = src->data;
    float *dst_data = dst->data;

    long n_jobs = 3 * sysconf(_SC_NPROCESSORS_ONLN);
    tile_job_t *jobs = malloc(sizeof(tile_job_t) * n_jobs);

    unsigned int n_y_tiles = CEIL_DIV(fill_height, tile_height);
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

    int tiles_x = CEIL_DIV(src_width, tile_width);
    int tiles_y = CEIL_DIV(src_height, tile_height);

    int dst_height = tiles_x * tiles_y;
    int dst_width = tile_height * tile_width;
    tensor *dst = tensor_init_2d(dst_height, dst_width);

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
