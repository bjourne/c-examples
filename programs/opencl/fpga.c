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
#include "tensors/tiling.h"

#include "matmul_fpga_config.h"

typedef enum {
    BUF_A,
    BUF_B,
    BUF_C
} buf_type;

int
main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Usage: %s platform-index kernel-path N M K\n", argv[0]);
        exit(1);
    }
    cl_uint N = atoi(argv[3]);
    cl_uint M = atoi(argv[4]);
    cl_uint K = atoi(argv[5]);

    int A_Y = N * A_BLOCK_Y;
    int A_X = M * A_BLOCK_X;
    int B_Y = A_X;
    int B_X = K * B_BLOCK_X;
    int C_Y = A_Y;
    int C_X = B_X;

    // Type sizes
    size_t n_uint = sizeof(cl_uint);
    size_t n_ulong = sizeof(cl_ulong);
    size_t n_mem = sizeof(cl_mem);
    size_t n_float = sizeof(float);

    tensor *a = tensor_init_2d(A_Y, A_X);
    tensor *b = tensor_init_2d(B_Y, B_X);
    tensor *c = tensor_init_2d(C_Y, C_X);

    printf("** Matrix dimensions **\n");
    printf("  %-10s %6d %6d\n", "a", A_Y, A_X);
    printf("  %-10s %6d %6d\n", "b", B_Y, B_X);
    printf("  %-10s %6d %6d\n", "c", C_Y, C_X);
    printf("\n");
    printf("** Kernel setup **\n");
    printf("  %-10s %4d\n", "X scale", X_SCALE);
    printf("  %-10s %4d %4d\n", "PE dims", PE_Y, PE_X);
    printf("  %-10s %4d %4d\n", "Block A", A_BLOCK_Y, A_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Block B", B_BLOCK_Y, B_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Block C", C_BLOCK_Y, C_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Interleave", Y_ILEAVE, X_ILEAVE);
    printf("\n");

    printf("** Initializing input matrices **\n");
    tensor_fill_rand_range(a, 20);
    tensor_fill_rand_range(b, 20);
    tensor_unary(a, a, TENSOR_UNARY_OP_ADD, -10.0);
    tensor_unary(b, b, TENSOR_UNARY_OP_ADD, -10.0);

    printf("** Multiplying on CPU **\n");
    tensor *c_ref = tensor_multiply_new(a, b);

    // Can we do this in one step?
    tensor *c_ref_tiled = tensor_tile_2d_new(
        c_ref, A_BLOCK_Y, B_BLOCK_X, 0, 0
    );

    int n_tiles = N * K * A_BLOCK_Y;

    tensor *c_ref_tiled_transposed = tensor_init_3d(n_tiles, PE_X, X_ILEAVE);

    tensor_set_dims(c_ref_tiled, 3, (int[]){n_tiles, X_ILEAVE, PE_X});

    tensor_transpose_tiled(c_ref_tiled, c_ref_tiled_transposed);

    tensor *a_tiled = tensor_tile_2d_new(
        a, A_BLOCK_Y, A_BLOCK_X, 0, 0
    );

    tensor *b_transpose = tensor_transpose_new(b);
    tensor *b_transpose_tiled = tensor_tile_2d_new(
        b_transpose, B_BLOCK_X, B_BLOCK_Y, 0, 0
    );

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

    size_t n_bytes_a = A_Y * A_X * n_float;
    size_t n_bytes_b = B_Y * B_X * n_float;
    size_t n_bytes_c = C_Y * C_X * n_float;

    ocl_ctx_buf bufs[3] = {
        {0, n_bytes_a, CL_MEM_READ_ONLY},
        {0, n_bytes_b, CL_MEM_READ_ONLY},
        {0, n_bytes_c, CL_MEM_WRITE_ONLY}
    };
    for (int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, bufs[i]));
    }
    OCL_CHECK_ERR(ocl_ctx_write_buffer(ctx, 0, BUF_A, a_tiled->data));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
        ctx, 1, BUF_B, b_transpose_tiled->data));
    ocl_ctx_arg kern_a_args[] = {
        {n_mem, &ctx->buffers[BUF_A].ptr},
        {n_uint, &M},
        {n_uint, &N},
        {n_uint, &K}
    };
    ocl_ctx_arg kern_b_args[] = {
        {n_mem, &ctx->buffers[BUF_B].ptr},
        {n_uint, &M},
        {n_uint, &N},
        {n_uint, &K}
    };
    ocl_ctx_arg kern_store_args[] = {
        {n_mem, &ctx->buffers[BUF_C].ptr},
        {n_uint, &N},
        {n_uint, &K}
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
    tensor_check_equal_contents(c, c_ref_tiled_transposed, 1.0);

    ocl_ctx_free(ctx);

    // Free tensors
    tensor_free(a);
    tensor_free(a_tiled);
    tensor_free(b);
    tensor_free(b_transpose);
    tensor_free(b_transpose_tiled);
    tensor_free(c);
    tensor_free(c_ref);
    tensor_free(c_ref_tiled);
    tensor_free(c_ref_tiled_transposed);
}
