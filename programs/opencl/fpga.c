// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"
#include "files/files.h"
#include "linalg/linalg.h"
#include "opencl/opencl.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"

#define HOST
#include "matmul_fpga_config.h"

#define SCALING_FACTOR  8L

// Matrix A height and width
#define HA (8L * A_BLOCK_Y)
#define WA (SCALING_FACTOR * A_BLOCK_X)

// Matrix B height and width
#define HB WA
#define WB (4L * B_BLOCK_X)

// Matrix C height and width
#define HC HA
#define WC WB

#define A_N_BLOCKS_X             (WA / A_BLOCK_X)
#define A_N_BLOCKS_Y             (HA / A_BLOCK_Y)
#define A_N_VECTORS_IN_ROW_OF_BLOCKS  (A_N_BLOCKS_X * MAT_A_BLOCK_NUM_VECTORS)

#define B_N_BLOCKS_X                        (WB / B_BLOCK_X)
#define B_N_BLOCKS_Y                        (HB / B_BLOCK_Y)
#define MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS  (B_N_BLOCKS_Y * MAT_B_BLOCK_NUM_VECTORS)
#define MAT_B_NUM_VECTORS_IN_MATRIX         (MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS * B_N_BLOCKS_X)

// Gold stuff
#define COMPUTE_GOLD_BLOCK_SIZE 64

#define HA_trim (2 *  A_BLOCK_Y)
#define WB_trim WB

static void
reorder_within_blocks(float * src,
                      float * dst,
                      int height, int width,
                      int num_sys_arr_columns, int block_x) {
    int n_els = height * width;
    int column_interleaving = block_x / num_sys_arr_columns;
    int word_id = 0;
    for (int i = 0; i < n_els; i += block_x) {
        for (int j = 0; j < column_interleaving; j++) {
            for (int k = 0; k < num_sys_arr_columns ; k++) {
                dst[word_id] = src[i + j + k * column_interleaving];
                word_id++;
            }
        }
    }
    assert(word_id == n_els);
}

typedef enum {
    BUF_A,
    BUF_B,
    BUF_C
} buf_type;

int
main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s platform-index kernel-path\n", argv[0]);
        exit(1);
    }

    // Type sizes
    size_t n_uint = sizeof(cl_uint);
    size_t n_uchar = sizeof(cl_uchar);
    size_t n_ulong = sizeof(cl_ulong);
    size_t n_mem = sizeof(cl_mem);
    size_t n_float = sizeof(float);

    tensor *a = tensor_init_2d(HA, WA);
    tensor *a_blocked = tensor_init_2d(HA, WA);
    tensor *b_transpose = tensor_init_2d(WB, HB);
    tensor *b_transpose_blocked = tensor_init_2d(WB, HB);
    tensor *b = tensor_init_2d(HB, WB);

    tensor *c = tensor_init_2d(HC, WC);
    tensor *c_golden = tensor_init_2d(HC, WC);
    tensor *c_golden_blocked = tensor_init_2d(HC, WC);
    tensor *c_golden_blocked_reordered = tensor_init_2d(HC, WC);
    tensor *c_ref = tensor_init_2d(HC, WC);

    printf("** Matrix dimensions **\n");
    printf("%12s %6ld %6ld\n", "a", HA, WA);
    printf("%12s %6ld %6ld\n", "b", HB, WB);
    printf("%12s %6ld %6ld\n", "c", HC, WC);
    printf("%12s %6ld %6ld\n", "a_blocked", HA, WA);
    printf("%12s %6ld %6ld\n", "b_transpose", WB, HB);
    printf("\n");
    printf("** Kernel setup **\n");
    printf("%12s %4d %4d\n", "PE dims", PE_ROWS, PE_COLS);
    printf("%12s %4d %4d\n", "Block A", A_BLOCK_Y, A_BLOCK_X);
    printf("%12s %4d %4d\n", "Block B", B_BLOCK_Y, B_BLOCK_X);
    printf("%12s %4d %4d\n", "Block C", C_BLOCK_Y, C_BLOCK_X);
    printf("%12s %4d %4d\n", "Interleave", ROWS_INTERLEAVED, COLUMNS_INTERLEAVED);
    printf("\n");

    printf("** Initializing input matrices **\n");
    tensor_fill_rand_range(a, 20);
    tensor_fill_rand_range(b, 20);
    tensor_unary(a, a, TENSOR_UNARY_OP_ADD, -10.0);
    tensor_unary(b, b, TENSOR_UNARY_OP_ADD, -10.0);

    printf("** Multiplying on CPU **\n");
    tensor_multiply(a, b, c_ref);

    tensor_linearize_tiles(a, a_blocked, A_BLOCK_Y, A_BLOCK_X);
    tensor_transpose(b, b_transpose);
    tensor_linearize_tiles(b_transpose, b_transpose_blocked,
                           B_BLOCK_X, B_BLOCK_Y);

    // Golden compute
    tensor_multiply(a, b, c_golden);
    tensor_linearize_tiles(c_golden, c_golden_blocked,
                           C_BLOCK_Y, C_BLOCK_X);
    reorder_within_blocks(c_golden_blocked->data,
                          c_golden_blocked_reordered->data,
                          HC, WC,
                          PE_COLS,
                          C_BLOCK_X);

    printf("** Setting up OpenCL **\n");

    int plat_idx = atoi(argv[1]);
    ocl_ctx *ctx = ocl_ctx_init(plat_idx, 0, true);

    OCL_CHECK_ERR(ocl_ctx_load_kernels(
        ctx,
        argv[2], "-cl-std=CL2.0 -Werror",
        3, (char *[]){"loadA", "loadB", "store"}
    ));

    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    for (int i = 0; i < 4; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_queue(ctx, props));
    }

    size_t n_bytes_a = HA * WA * n_float;
    size_t n_bytes_b = HB * WB * n_float;
    size_t n_bytes_c = HC * WC * n_float;

    ocl_ctx_buf bufs[3] = {
        {0, n_bytes_a, CL_MEM_READ_ONLY},
        {0, n_bytes_b, CL_MEM_READ_ONLY},
        {0, n_bytes_c, CL_MEM_WRITE_ONLY}
    };
    for (int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, bufs[i]));
    }
    OCL_CHECK_ERR(ocl_ctx_write_buffer(ctx, 0, BUF_A, a_blocked->data));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(ctx, 1, BUF_B, b_transpose_blocked->data));

    // LoadA kernel
    unsigned int mat_a_num_vectors_in_row_of_blocks =
        A_N_VECTORS_IN_ROW_OF_BLOCKS;

    unsigned char a_n_blocks_y = A_N_BLOCKS_Y;
    unsigned char b_n_blocks_x = B_N_BLOCKS_X;

    // LoadB kernel
    unsigned int mat_b_num_vectors_in_col_of_blocks =
        MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS;
    unsigned int mat_b_num_vectors_in_matrix =
        MAT_B_NUM_VECTORS_IN_MATRIX;

    // Store kernel
    int mat_c_num_coalesced_words = WC * HC / PE_COLS;
    ocl_ctx_arg kern_a_args[] = {
        {n_mem, &ctx->buffers[BUF_A].ptr},
        {n_uint, &mat_a_num_vectors_in_row_of_blocks},
        {n_uchar, &a_n_blocks_y},
        {n_uchar, &b_n_blocks_x}
    };
    ocl_ctx_arg kern_b_args[] = {
        {n_mem, &ctx->buffers[BUF_B].ptr},
        {n_uint, &mat_b_num_vectors_in_col_of_blocks},
        {n_uint, &mat_b_num_vectors_in_matrix},
        {n_uchar, &a_n_blocks_y}
    };
    ocl_ctx_arg kern_store_args[] = {
        {n_mem, &ctx->buffers[BUF_C].ptr},
        {n_uint, &mat_c_num_coalesced_words}
    };
    OCL_CHECK_ERR(ocl_ctx_set_kernels_arguments(
        ctx,
        ARRAY_SIZE(kern_a_args), kern_a_args,
        ARRAY_SIZE(kern_b_args), kern_b_args,
        ARRAY_SIZE(kern_store_args), kern_store_args
    ));

    // Queue kernels
    size_t local[] = {1};
    size_t global[] = {1};
    cl_event events[3];
    for (int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(clEnqueueNDRangeKernel(
            ctx->queues[i], ctx->kernels[i],
            1, NULL, global, local,
            0, NULL,
            &events[i]
        ));
    }
    for(int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(clFlush(ctx->queues[i]));
        OCL_CHECK_ERR(clFinish(ctx->queues[i]));
    }

    // Compute execution time
    for (int i = 0; i < 3; i++) {
        cl_ulong start, end;
        OCL_CHECK_ERR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                              n_ulong, &end, NULL));
        OCL_CHECK_ERR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                              n_ulong, &start, NULL));
        double time = 1.0e-9 * (end - start);
        printf("%.6f\n", time);
    }

    // We use the fourth queue to read data back.
    OCL_CHECK_ERR(ocl_ctx_read_buffer(ctx, 3, BUF_C, c->data));
    tensor_check_equal(c, c_golden_blocked_reordered, 1.0);

    ocl_ctx_free(ctx);

    // Free tensors
    tensor_free(a);
    tensor_free(a_blocked);
    tensor_free(b);
    tensor_free(b_transpose);
    tensor_free(b_transpose_blocked);
    tensor_free(c);
    tensor_free(c_ref);
    tensor_free(c_golden);
    tensor_free(c_golden_blocked);
    tensor_free(c_golden_blocked_reordered);
}
