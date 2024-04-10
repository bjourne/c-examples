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

#define SCALING_FACTOR  128

#define HA (128 * MAT_A_BLOCK_HEIGHT)           // Matrix A height
#define WA (SCALING_FACTOR * MAT_A_BLOCK_WIDTH) // Matrix A width

#define HB WA                                   // Matrix B height
#define WB (128 * MAT_B_BLOCK_WIDTH)            // Matrix B width

#define HC HA                                   // Matrix C height
#define WC WB                                   // Matrix C width


// A+B+C matrices = 1.47 GB (with scaling factor 24)
// S10 EA board DDR memory = 2 GB


#define MAT_A_N_BLOCKS_IN_ROW             (WA / MAT_A_BLOCK_WIDTH)
#define MAT_A_N_BLOCKS_IN_COL             (HA / MAT_A_BLOCK_HEIGHT)
#define MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS  (MAT_A_N_BLOCKS_IN_ROW * MAT_A_BLOCK_NUM_VECTORS)

#define MAT_B_N_BLOCKS_IN_ROW             (WB / MAT_B_BLOCK_WIDTH)
#define MAT_B_N_BLOCKS_IN_COL             (HB / MAT_B_BLOCK_HEIGHT)
#define MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS  (MAT_B_N_BLOCKS_IN_COL * MAT_B_BLOCK_NUM_VECTORS)
#define MAT_B_NUM_VECTORS_IN_MATRIX         (MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS * MAT_B_N_BLOCKS_IN_ROW)


// Gold stuff
#define COMPUTE_GOLD_BLOCK_SIZE 64

#define HA_trim (2 *  MAT_A_BLOCK_HEIGHT)
#define WB_trim WB

static  void
reorder_within_blocks(float * src,
                      float * dst,
                      int height, int width,
                      int num_sys_arr_columns, int block_width) {
    int n_els = height * width;
    int column_interleaving = block_width / num_sys_arr_columns;
    int word_id = 0;
    for (int i = 0; i < n_els; i += block_width) {
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
    BUF_A = 0,
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

    tensor *a = tensor_init(2, (int[]){HA, WA});
    tensor *a_blocked = tensor_init(2, (int[]){HA, WA});
    tensor *b_transpose = tensor_init(2, (int[]){WB, HB});
    tensor *b_transpose_blocked = tensor_init(2, (int[]){WB, HB});
    tensor *b = tensor_init(2, (int[]){HB, WB});

    tensor *c = tensor_init(2, (int[]){HC, WC});
    tensor *c_golden = tensor_init(2, (int[]){HC, WC});
    tensor *c_golden_blocked = tensor_init(2, (int[]){HC, WC});
    tensor *c_golden_blocked_reordered = tensor_init(2, (int[]){HC, WC});
    tensor *c_ref = tensor_init(2, (int[]){HC, WC});

    printf("** Matrix dimensions **\n");
    printf("%12s %4d %4d\n", "a", HA, WA);
    printf("%12s %4d %4d\n", "b", HB, WB);
    printf("%12s %4d %4d\n", "c", HC, WC);
    printf("%12s %4d %4d\n", "a_blocked", HA, WA);
    printf("%12s %4d %4d\n", "b_transpose", WB, HB);
    printf("\n");
    printf("** Kernel setup **\n");
    printf("%12s %4d %4d\n", "PE dims", PE_ROWS, PE_COLS);
    printf("%12s %4d %4d\n", "Block A", MAT_A_BLOCK_HEIGHT, MAT_A_BLOCK_WIDTH);
    printf("%12s %4d %4d\n", "Block B", MAT_B_BLOCK_HEIGHT, MAT_B_BLOCK_WIDTH);
    printf("%12s %4d %4d\n", "Block C", MAT_C_BLOCK_HEIGHT, MAT_C_BLOCK_WIDTH);
    printf("%12s %4d %4d\n", "Interleave", ROWS_INTERLEAVED, COLUMNS_INTERLEAVED);
    printf("\n");

    printf("** Initializing input matrices **\n");
    tensor_fill_rand_range(a, 20);
    tensor_fill_rand_range(b, 20);
    tensor_unary(a, a, TENSOR_UNARY_OP_ADD, -10.0);
    tensor_unary(b, b, TENSOR_UNARY_OP_ADD, -10.0);
    tensor_fill_const(c, 0);

    printf("** Multiplying on CPU**\n");
    tensor_multiply(a, b, c_ref);

    tensor_linearize_tiles(a, a_blocked, MAT_A_BLOCK_HEIGHT, MAT_A_BLOCK_WIDTH);
    tensor_transpose(b, b_transpose);
    tensor_linearize_tiles(b_transpose, b_transpose_blocked,
                           MAT_B_BLOCK_WIDTH, MAT_B_BLOCK_HEIGHT);

    // Golden compute
    tensor_multiply(a, b, c_golden);
    tensor_linearize_tiles(c_golden, c_golden_blocked,
                           MAT_C_BLOCK_HEIGHT, MAT_C_BLOCK_WIDTH);
    reorder_within_blocks(c_golden_blocked->data,
                          c_golden_blocked_reordered->data,
                          HC, WC,
                          PE_COLS,
                          MAT_C_BLOCK_WIDTH);

    printf("** Setting up OpenCL **\n");

    int plat_idx = atoi(argv[1]);
    ocl_ctx *ctx = ocl_ctx_init(plat_idx, 0, true);

    printf("Loading kernels\n");
    OCL_CHECK_ERR(ocl_ctx_load_kernels(
                      ctx,
                      argv[2], "-cl-std=CL2.0 -Werror",
                      3, (char *[]){"loadA", "loadB", "store"}));

    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    for (int i = 0; i < 4; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_queue(ctx, props));
    }

    printf("Creating device buffers\n");
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
    printf("Initializing LoadA kernel\n");
    unsigned int mat_a_num_vectors_in_row_of_blocks =
        MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS;
    unsigned char mat_a_num_blocks_in_col = MAT_A_N_BLOCKS_IN_COL;
    unsigned char mat_b_num_blocks_in_row = MAT_B_N_BLOCKS_IN_ROW;

    // LoadB kernel
    printf("Initializing LoadB kernel\n");
    unsigned int mat_b_num_vectors_in_col_of_blocks =
        MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS;
    unsigned int mat_b_num_vectors_in_matrix =
        MAT_B_NUM_VECTORS_IN_MATRIX;

    // Store kernel
    int mat_c_num_coalesced_words = WC * HC / PE_COLS;
    ocl_ctx_arg kern_a_args[] = {
        {n_mem, &ctx->buffers[BUF_A].ptr},
        {n_uint, &mat_a_num_vectors_in_row_of_blocks},
        {n_uchar, &mat_a_num_blocks_in_col},
        {n_uchar, &mat_b_num_blocks_in_row}
    };
    ocl_ctx_arg kern_b_args[] = {
        {n_mem, &ctx->buffers[BUF_B].ptr},
        {n_uint, &mat_b_num_vectors_in_col_of_blocks},
        {n_uint, &mat_b_num_vectors_in_matrix},
        {n_uchar, &mat_a_num_blocks_in_col}
    };
    ocl_ctx_arg kern_store_args[] = {
        {n_mem, &ctx->buffers[BUF_C].ptr},
        {n_uint, &mat_c_num_coalesced_words}
    };
    OCL_CHECK_ERR(ocl_ctx_set_kernels_arguments(
                      ctx,
                      ARRAY_SIZE(kern_a_args), kern_a_args,
                      ARRAY_SIZE(kern_b_args), kern_b_args,
                      ARRAY_SIZE(kern_store_args), kern_store_args));

    // Queue kernels
    size_t local[] = {1};
    size_t global[] = {1};
    cl_event events[3];
    for (int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(clEnqueueNDRangeKernel(
                          ctx->queues[i], ctx->kernels[i],
                          1, NULL, global, local,
                          0, NULL,
                          &events[i]));
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
