// Copyright (C) 2022-2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
//
// Here are performance figures for the Agilex 7 FPGA I'm working
// with. For N=M=K=8192 matrices:
//
// | VSIZE | PE    | ILEAV | LVEC | SEED | SW  | LAT | FN   | FMAX | TIME  |
// |-------|-------|-------|------|------|-----|-----|------|------|-------|
// | 8     | 8x8   | 16x16 | 1    |      | 16  |     |      |      | 4.93  |
// | 8     | 16x16 | 16x16 | 1    |      | 16  |     |      | 445  | 2.07  |
// | 8     | 16x16 | 16x16 | 1    | 9999 | 16  |     |      | 442  | 2.08  |
// | 8     | 16x16 | 16x16 | 1    | 9998 | 16  |     |      | 460  | 1.32  |
// | 8     | 16x16 | 16x16 | 1    | 9997 | 16  |     |      | 438  | 1.37  |
// | 8     | 16x16 | 16x16 | 1    | 9996 | 16  |     |      | 425  | 1.42  |
// | 8     | 16x16 | 16x16 | 1    | 9995 | 16  |     |      | 431  | 1.39  |
// | 8     | 16x16 | 16x16 | 2    | 9994 | 16  |     |      | 406  | 0.90  |
// | 8     | 16x16 | 16x16 | 4    | 9993 | 16  |     |      | 461  | 32.63 |
// | 8     | 16x16 | 16x16 | 2    | 9992 | 16  |     | (1)  | 456  | 1.24  |
// | 8     | 16x16 | 16x16 | 2    | 9991 | 16  |     | (2)  | 605  | 0.57  |
// | 8     | 16x16 | 16x16 | 2    | 9990 | 128 |     |      | 500  | 0.91  |
// | 8     | 16x16 | 16x16 | 2    | 9989 | 64  |     |      | 500  | 0.89  |
// | 8     | 16x16 | 16x16 | 2    | 9988 | 16  |     |      | 585  | 0.70  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     |      | 485  | 0.82  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     | (3)  | 550  | 0.73  |
// | 8     | 16x16 | 16x16 | 2    | 9987 | 16  |     | (4)  | 492  | 0.74  |
// | 8     | 16x16 | 16x16 | 2    | 9986 | 16  |     | (4)  | 565  | 0.74  |
// | 8     | 16x16 | 16x16 | 2    | 9985 | 16  |     | (5)  | 565  | 0.58  |
// | 8     | 16x16 | 16x16 | 2    | 9985 | 16  |     | (6)  | -    | -     |
// | 8     | 16x16 | 16x16 | 2    | 9984 | 16  |     |      | 595  | 0.58  |
// | 8     | 16x16 | 16x16 | 2    | 9984 | 16  |     | (7)  | 606  | -     |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  |     | (8)  | 600  | 0.57  |
// | 8     | 16x16 | 16x16 | 4    | 9983 | 16  |     |      | 508  | 28.54 |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  |     | (9)  | 538  | 0.61  |
// | 8     | 16x16 | 16x16 | 2    | 9983 | 16  |     | (10) | 603  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9982 | 16  |     |      | 606  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9982 | 16  |     |      | 600  | 0.55  |
// | 8     | 16x16 | 16x16 | 2    | 9981 | 16  |     | (11) | 560  | 0.65  |
// | 8     | 16x16 | 16x16 | 2    | 9981 | 16  |     | (12) | 592  | 0.46  |
// | 8     | 32x16 | 32x16 | 2    | 9981 | 16  |     | (13) | 605  | -     |
// | 4     | 16x16 | 16x16 | 2    | 9981 | -   |     | (14) |      |       |
// | 8     | 16x16 | 16x16 | 2    | 9979 | 16  |     |      | 610  | 0.46  |
// | 8     | 16x16 | 16x16 | 2    | 9979 | 16  |     | (15) | 608  | 0.45  |
// | 8     | 16x16 | 16x16 | 4    | 9978 | 16  |     |      | 567  | 0.47  |
// | 8     | 16x16 | 16x16 | 1    | 9978 | 16  |     |      | 600  | 0.47  |
// | 4     | 16x16 | 16x16 | 1    | 9977 | 16  |     |      | 608  | 0.93  |
// | 8     | 16x16 | 16x16 | -    | 9999 | 16  | 89  | (16) | 563  | 0.50  |
// | 8     | 16    | -     | -    | 9970 | 16  | 89  | (17) | 575  | 0.49  |
// | 8     | 16    | -     | -    | 9969 | 16  | 89  | (18) | 588  | 0.48  |
// | 8     | 16    | -     | -    | 9968 | 16  | 89  |      | 585  | 0.48  |
// | 8     | 16    | -     | -    | 9968 | 16  | 89  |      | 585  | 0.48  |
// | 8     | 16    | -     | -    | 9963 | 16  | 88  | (19) | 600  | 0.47  |
//
// LAT = Latency of innermost loop
// SW = I forgot
//
// 1. This refactoring increased the length of the critical chain.
// 2. Reverted last changes.
// 3. No volatile store
// 4. Simpler store kernel
// 5. No volatile
// 6. No FPGA_REGx (breaks Quartus)
// 7. Removed some FPGA_REG2 (causes incorrect results)
// 8. -cl-fast-relaxed-math -cl-mad-enable
// 9. Channel depth 512
// 10. Channel depth 256
// 11. X_SCALE=32
// 12. X_SCALE=8
// 13. Breaks code
// 14. Breaks Quartus
// 15. No interleaving
// 16. No LVEC
// 17. Square SAs
// 18. Simpler counter mgmt
// 19. Simpler handling of the clear signal
//
// This is important but it is not enforced (hmmm):
// PE_X + PE_Y <= Y_INTERLEAVED

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

#define CL_CHANNEL_1_INTELFPGA              (1 << 16)
#define CL_CHANNEL_2_INTELFPGA              (2 << 16)
#define CL_CHANNEL_3_INTELFPGA              (3 << 16)
#define N_KERNELS                           3

typedef enum {
    BUF_A,
    BUF_B,
    BUF_C
} buf_type;

static void
usage(char *bin) {
    printf(
        "Usage: %s "
        "platform-index kernel-path "
        "V_SIZE PE_S X_SCALE "
        "N M K\n\n",
        bin
    );
    printf("    V_SIZE      SIMD vector size (1, 2, 4, 8, or 16)\n");
    printf("    PE_S        Side length of the square systolic array\n");
    printf("    X_SCALE     Scaling factor\n");
    printf("    N, M, K\n");
    printf("        Input matrix sizes NxM and MxK in blocks\n");
    exit(1);
}

int
main(int argc, char *argv[]) {
    if (argc != 9) {
        usage(argv[0]);
    }
    // Systolic array setup
    cl_uint V_SIZE = atoi(argv[3]);
    cl_uint PE_S = atoi(argv[4]);
    cl_uint X_SCALE = atoi(argv[5]);

    // Matrix dimensions in blocks
    cl_uint N = atoi(argv[6]);
    cl_uint M = atoi(argv[7]);
    cl_uint K = atoi(argv[8]);

    cl_uint A_BLOCK_Y = PE_S * PE_S;
    cl_uint A_BLOCK_X = X_SCALE * V_SIZE;

    cl_uint B_BLOCK_Y = A_BLOCK_X;
    cl_uint B_BLOCK_X = A_BLOCK_Y;

    cl_uint C_BLOCK_Y = A_BLOCK_Y;
    cl_uint C_BLOCK_X = B_BLOCK_X;

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
    printf("  %-10s %4d %4d\n", "PE dims", PE_S, PE_S);
    printf("  %-10s %4d %4d\n", "Block A", A_BLOCK_Y, A_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Block B", B_BLOCK_Y, B_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Block C", C_BLOCK_Y, C_BLOCK_X);
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

    tensor *c_ref_tiled_transposed = tensor_init_3d(n_tiles, PE_S, PE_S);

    tensor_set_dims(c_ref_tiled, 3, (int[]){n_tiles, PE_S, PE_S});

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
    ocl_ctx *ctx = ocl_ctx_init(plat_idx, 0, false);

    char opts[256];
    sprintf(
        opts,
        "-cl-std=CL2.0 -Werror -D PE_S=%d -D V_SIZE=%d -D X_SCALE=%d",
        PE_S, V_SIZE, X_SCALE
    );

    char *kernels[] = {"load_a", "load_b", "store"};
    OCL_CHECK_ERR(ocl_ctx_load_kernels(
        ctx,
        argv[2], opts,
        N_KERNELS, kernels
    ));

    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    for (int i = 0; i < N_KERNELS; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_queue(ctx, props));
    }

    size_t n_bytes_a = A_Y * A_X * n_float;
    size_t n_bytes_b = B_Y * B_X * n_float;
    size_t n_bytes_c = C_Y * C_X * n_float;

    ocl_ctx_buf bufs[3] = {
        {0, n_bytes_a, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA},
        {0, n_bytes_b, CL_MEM_READ_ONLY | CL_CHANNEL_2_INTELFPGA},
        {0, n_bytes_c, CL_MEM_WRITE_ONLY | CL_CHANNEL_3_INTELFPGA}
    };
    for (int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, bufs[i]));
    }
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
        ctx, 0, BUF_A, a_tiled->data
    ));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
        ctx, 0, BUF_B, b_transpose_tiled->data
    ));
    ocl_ctx_arg kern_a_args[] = {
        {n_mem, &ctx->buffers[BUF_A].ptr},
        {n_uint, &N},
        {n_uint, &M},
        {n_uint, &K}
    };
    ocl_ctx_arg kern_b_args[] = {
        {n_mem, &ctx->buffers[BUF_B].ptr},
        {n_uint, &N},
        {n_uint, &M},
        {n_uint, &K}
    };
    ocl_ctx_arg kern_store_args[] = {
        {n_mem, &ctx->buffers[BUF_C].ptr},
        {n_uint, &N},
        {n_uint, &M},
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
    cl_event events[N_KERNELS];
    for (int i = 0; i < N_KERNELS; i++) {
        OCL_CHECK_ERR(clEnqueueNDRangeKernel(
            ctx->queues[i], ctx->kernels[i],
            1, NULL, global, local,
            0, NULL,
            &events[i]
        ));
    }
    for(int i = 0; i < N_KERNELS; i++) {
        OCL_CHECK_ERR(clFlush(ctx->queues[i]));
        OCL_CHECK_ERR(clFinish(ctx->queues[i]));
    }

    // Compute execution time
    for (int i = 0; i < N_KERNELS; i++) {
        cl_ulong start, end;
        OCL_CHECK_ERR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                              n_ulong, &end, NULL));
        OCL_CHECK_ERR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                              n_ulong, &start, NULL));
        double time = 1.0e-9 * (end - start);
        printf("%-15s: %.4f s\n", kernels[i], time);
    }

    // Use first queue to read data back.
    OCL_CHECK_ERR(ocl_ctx_read_buffer(ctx, 0, BUF_C, c->data));
    tensor_check_equal_contents(c, c_ref_tiled_transposed, 0.1);

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
