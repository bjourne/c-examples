// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <time.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"
#include "tensors/tiling.h"

void
test_reorder() {
    int n_tiles = 5;
    int tile_x = 3;
    int tile_y = 4;

    tensor *src = tensor_init_3d(n_tiles, tile_y, tile_x);
    tensor *dst = tensor_init_3d(n_tiles, tile_x, tile_y);
    tensor_fill_range(src, 0.0);

    tensor_transpose_tiled(src, dst);

    assert(src->dims[1] == dst->dims[2]);
    assert(src->dims[2] == dst->dims[1]);

    tensor_print(src, true, 0, 200, " ");
    tensor_print(dst, true, 0, 200, " ");

    tensor_free(src);
    tensor_free(dst);
}

void
test_tile_2d_mt_small() {
    int sy = 7;
    int sx = 7;
    int ty = 2;
    int tx = 2;

    tensor *src = tensor_init_2d(sy, sx);
    tensor_fill_range(src, 1.0);
    tensor *tiled0 = tensor_tile_2d_new(src, ty, tx, 0, 0);
    tensor *tiled1 = tensor_tile_2d_mt_new(src, ty, tx, 0, 0);

    tensor_check_equal(tiled0, tiled1, LINALG_EPSILON);
    tensor_free(src);
    tensor_free(tiled0);
    tensor_free(tiled1);
}

void
test_tile_2d_mt() {
    int sy = 200 + rand_n(100);
    int sx = 200 + rand_n(100);
    int ty = 4 + rand_n(4);
    int tx = 4 + rand_n(4);

    tensor *src = tensor_init_2d(sy, sx);
    tensor_fill_range(src, 1.0);
    tensor *tiled0 = tensor_tile_2d_new(src, ty, tx, 0, 0);
    tensor *tiled1 = tensor_tile_2d_mt_new(src, ty, tx, 0, 0);
    tensor_check_equal(tiled0, tiled1, LINALG_EPSILON);
    tensor_free(src);
    tensor_free(tiled0);
    tensor_free(tiled1);
}

void
test_linearize_tiles() {
    tensor *src = tensor_init_2d(4, 4);
    tensor *dst = tensor_init_4d(2, 2, 2, 2);
    tensor *dst_ref = tensor_init_4d(2, 2, 2, 2);
    tensor_copy_data(dst_ref, (float[]){
        1, 2, 5, 6,
        3, 4, 7, 8,
        9, 10, 13, 14,
        11, 12, 15, 16});

    tensor_fill_range(src, 1.0);

    tensor_tile_2d(src, dst);
    tensor_check_equal(dst, dst_ref, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst);
    tensor_free(dst_ref);
}

void
test_transpose_a() {
    tensor *src = tensor_init_2d(5, 4);
    tensor_fill_range(src, 1.0);

    tensor *dst1 = tensor_tile_2d_new(src, 2, 1, 0, 0);
    assert(dst1->dims[0] == 3);
    assert(dst1->dims[1] == 4);
    assert(dst1->dims[2] == 2);
    assert(dst1->dims[3] == 1);
    tensor_free(dst1);

    tensor *dst2 = tensor_tile_2d_new(src, 3, 1, 0, 0);
    assert(tensor_n_elements(dst2) == 2 * 4 * 3 * 1);
    tensor_free(dst2);

    tensor *dst3 = tensor_tile_2d_new(src, 4, 1, 0, 0);
    assert(dst3->n_dims == 4);
    assert(tensor_n_elements(dst3) == 2 * 4 * 4 * 1);
    tensor_free(dst3);

    tensor *dst4 = tensor_tile_2d_new(src, 5, 1, 0, 0);
    assert(tensor_n_elements(dst4) == 1 * 4 * 5 * 1);
    tensor_free(dst4);

    tensor *dst5 = tensor_tile_2d_new(src, 6, 1, 0, 0);
    assert(tensor_n_elements(dst5) == 1 * 4 * 6 * 1);
    tensor_free(dst5);
    tensor_free(src);
}

void
test_transpose_b() {
    float matrix_6x4[6][4] = {
        {  1,   2,   5,   6},
        {  3,   4,   7,   8},
        {  9,  10,  13,  14},
        { 11,  12,  15,  16},
        { 17,  18,   0,   0},
        { 19,  20,   0,   0}
    };
    float matrix_6x6[6][6] = {
        {  1,   2,   3,   5,   6,   7},
        {  4,   0,   0,   8,   0,   0},
        {  9,  10,  11,  13,  14,  15},
        { 12,   0,   0,  16,   0,   0},
        { 17,  18,  19,   0,   0,   0},
        { 20,   0,   0,   0,   0,   0}
    };
    float matrix_4x9[4][9] = {
        {  1,   2,   3,   5,   6,   7,   9,  10,  11},
        {  4,   0,   0,   8,   0,   0,  12,   0,   0},
        { 13,  14,  15,  17,  18,  19,   0,   0,   0},
        { 16,   0,   0,  20,   0,   0,   0,   0,   0}
    };
    tensor *src = tensor_init_2d(5, 4);
    tensor_fill_range(src, 1.0);

    tensor *dst1 = tensor_tile_2d_new(src, 2, 2, 0, 0);
    tensor *dst1_exp = tensor_init_4d(3, 2, 2, 2);
    tensor_copy_data(dst1_exp, (float *)matrix_6x4);
    tensor_check_equal(dst1, dst1_exp, LINALG_EPSILON);

    tensor *dst2 = tensor_tile_2d_new(src, 2, 3, 0, 0);
    tensor *dst2_exp = tensor_init_4d(3, 2, 2, 3);
    tensor_copy_data(dst2_exp, (float *)matrix_6x6);
    tensor_check_equal(dst2, dst2_exp, LINALG_EPSILON);

    tensor *dst3 = tensor_tile_2d_new(src, 3, 3, 0, 0);
    tensor *dst3_exp = tensor_init_4d(2, 2, 3, 3);
    tensor_copy_data(dst3_exp, (float *)matrix_4x9);
    tensor_check_equal(dst3, dst3_exp, LINALG_EPSILON);

    tensor_free(src);
    tensor_free(dst1);
    tensor_free(dst1_exp);
    tensor_free(dst2);
    tensor_free(dst2_exp);
    tensor_free(dst3);
    tensor_free(dst3_exp);
}

void
test_tile_2d() {
    tensor *src = tensor_init_2d(4, 4);
    tensor_fill_range(src, 1.0);
    tensor *dst = tensor_tile_2d_new(src, 3, 3, 0, 0);

    assert(tensor_n_elements(dst) == 3 * 3 * 4);

    tensor_print(src, false, 0, 80, " ");
    tensor_print(dst, false, 0, 80, " ");
    tensor_free(dst);
    tensor_free(src);
}

void
test_filling() {
    tensor *src = tensor_init_2d(4, 4);
    tensor_fill_range(src, 1.0);
    tensor *dst = tensor_tile_2d_new(src, 2, 2, 8, 8);
    assert(tensor_n_elements(dst) == 4 * 4 * 2 * 2);
    tensor_free(src);
    tensor_free(dst);
}

void
perf_tile_2d() {
    int SIZE = 8192*2;
    tensor *src = tensor_init_2d(SIZE, SIZE);
    uint64_t bef = nano_count();
    for (int i = 0; i < 10; i++) {
        tensor *dst = tensor_tile_2d_new(src, 256, 64, SIZE, SIZE);
        tensor_free(dst);
    }
    double delta = (nano_count() - bef) / 1000000000.0;
    printf("%.6lfs\n", delta);
    tensor_free(src);
}

// 4.776 on gcc
// 4.800 on clang
void
perf_tile_2d_mt() {
    int SIZE = 8192*4;
    tensor *src = tensor_init_2d(SIZE, SIZE);
    uint64_t bef = nano_count();
    for (int i = 0; i < 10; i++) {
        tensor *dst = tensor_tile_2d_mt_new(src, 256, 64, SIZE, SIZE);
        tensor_free(dst);
    }
    double delta = (nano_count() - bef) / 1000000000.0;
    printf("%.6lfs\n", delta);
    tensor_free(src);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_reorder);
    PRINT_RUN(test_tile_2d_mt_small);
    PRINT_RUN(test_tile_2d_mt);
    PRINT_RUN(test_tile_2d);
    PRINT_RUN(test_linearize_tiles);
    PRINT_RUN(test_transpose_a);
    PRINT_RUN(test_transpose_b);
    PRINT_RUN(test_filling);
    perf_tile_2d();
    perf_tile_2d_mt();
}
