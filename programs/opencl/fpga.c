// Copyright (C) 2022-2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
//
// Here are performance figures for the Agilex 7 FPGA I'm working
// with. For N=M=K=8192 float matrices:
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
// | 8     | 16    | -     | -    | 9960 | 16  | 94  |      | 601  | 0.47  |
// | 8     | 16    | -     | -    | 9960 | 16  | 94  |      | 608  | 0.46  |
// | 8     | 16    | -     | -    | 9959 | 16  | 530 |      | 608  | 0.46  |
//
// For M=N=K=8192 char matrices:
//
//
// | VSIZE | PE    | ILEAV | LVEC | SEED | LAT | FN | FMAX | TIME |
// |-------|-------|-------|------|------|-----|----|------|------|
// | 16    | 16x16 | 16x16 | -    | 9959 | 519 |    | 600  | 0.23 |
//
//
// LAT = Latency of innermost loop
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
// 13-14. Breaks Quartus
// 15. No interleaving
// 16. No LVEC
// 17. Square SAs
// 18. Simpler counter mgmt
// 19. Simpler handling of the clear signal

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/bits.h"
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

cl_half
float_to_half(float f) {
    uint32_t bits = BW_FLOAT_TO_UINT(f);

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (bits & 0x007FFFFF) >> 13;

    if (exponent <= 0) {
        return sign;
    } else if (exponent >= 31) {
        return sign | 0x7C00;
    }
    return (cl_half)(sign | (exponent << 10) | mantissa);
}

float
half_to_float(cl_half h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF) << 13;

    if (exponent == 0) {
        return 0.0f;
    } else if (exponent == 31) {
        return BW_UINT_TO_FLOAT(sign | 0x7F800000 | mantissa);
    }

    exponent = (exponent - 15 + 127) << 23;
    uint32_t result = sign | exponent | mantissa;
    return BW_UINT_TO_FLOAT(result);
}

typedef enum {
    BUF_A,
    BUF_B,
    BUF_C
} buf_type;

typedef enum {
    V_TYPE_LONG = 1,
    V_TYPE_FLOAT,
    V_TYPE_INT,
    V_TYPE_HALF,
    V_TYPE_CHAR,
    V_TYPE_LAST
} v_type;

static int
V_TYPE_SIZE[V_TYPE_LAST] = {
    [V_TYPE_LONG] = sizeof(cl_long),
    [V_TYPE_FLOAT] = sizeof(cl_float),
    [V_TYPE_INT] = sizeof(cl_int),
    [V_TYPE_HALF] = sizeof(cl_half),
    [V_TYPE_CHAR] = sizeof(cl_char)
};

static bool
V_TYPE_IS_INTEGRAL[V_TYPE_LAST] = {
    [V_TYPE_LONG] = true,
    [V_TYPE_FLOAT] = false,
    [V_TYPE_INT] = true,
    [V_TYPE_HALF] = false,
    [V_TYPE_CHAR] = true
};

static void
matrix_init(tensor *t, v_type tp) {
    tensor_fill_rand_range(t, 20);
    tensor_unary(t, t, TENSOR_UNARY_OP_ADD, -10.0);
    if (V_TYPE_IS_INTEGRAL[tp]) {

    }
    tensor_unary(t, t, TENSOR_UNARY_OP_TRUNC, 0.0);
}




static void
scalar_copy_cast(void *dst, v_type dt, void *src, v_type st, int idx)   {
    double v;
    if (st == V_TYPE_LONG) {
        v = ((long *)src)[idx];
    } else if (st == V_TYPE_FLOAT) {
        v = ((float *)src)[idx];
    } else if (st == V_TYPE_INT) {
        v = ((int *)src)[idx];
    } else if (st == V_TYPE_HALF) {
        cl_half t = ((cl_half *)src)[idx];
        v = (double)half_to_float(t);
    } else if (st == V_TYPE_CHAR) {
        v = ((char *)src)[idx];
    } else {
        assert(false);
    }
    if (dt == V_TYPE_LONG) {
        ((long *)dst)[idx] = v;
    } else if (dt == V_TYPE_FLOAT) {
        ((float *)dst)[idx] = v;
    } else if (dt == V_TYPE_INT) {
        ((int *)dst)[idx] = v;
    } else if (dt == V_TYPE_HALF) {
        cl_half t = float_to_half(v);
        ((cl_half *)dst)[idx] = t;
    } else if (dt == V_TYPE_CHAR) {
        ((char *)dst)[idx] = v;
    } else {
        assert(false);
    }
}

static void
buffer_copy_cast(void *dst, v_type dt, void *src, v_type st, int n) {
    for (int i = 0; i < n; i++) {
        scalar_copy_cast(dst, dt, src, st, i);
    }
}

static void
usage(char *bin) {
    printf(
        "Usage: %s "
        "platform-index kernel-path "
        "V_TYPE V_SIZE PE_S X_SCALE "
        "N M K\n\n",
        bin
    );
    printf("    V_TYPE      SIMD vector type:\n");
    printf("                    1 for long\n");
    printf("                    2 for float\n");
    printf("                    3 for int\n");
    printf("                    4 for half\n");
    printf("                    5 for char\n");
    printf("    V_SIZE      SIMD vector size (1, 2, 4, 8, or 16)\n");
    printf("    PE_S        Side length of the square systolic array\n");
    printf("    X_SCALE     Scaling factor\n");
    printf("    N, M, K\n");
    printf("        Input matrix sizes NxM and MxK in blocks\n");
    exit(1);
}

int
main(int argc, char *argv[]) {
    if (argc != 10) {
        usage(argv[0]);
    }
    // Systolic array setup
    v_type v_type = atoi(argv[3]);
    int v_size = atoi(argv[4]);
    int pe_s = atoi(argv[5]);
    int x_scale = atoi(argv[6]);

    // Matrix dimensions in blocks
    int N = atoi(argv[7]);
    int M = atoi(argv[8]);
    int K = atoi(argv[9]);

    int A_BLOCK_Y = pe_s * pe_s;
    int A_BLOCK_X = x_scale * v_size;

    int B_BLOCK_Y = A_BLOCK_X;
    int B_BLOCK_X = A_BLOCK_Y;

    int C_BLOCK_Y = A_BLOCK_Y;
    int C_BLOCK_X = B_BLOCK_X;

    int A_Y = N * A_BLOCK_Y;
    int A_X = M * A_BLOCK_X;
    int B_Y = A_X;
    int B_X = K * B_BLOCK_X;
    int C_Y = A_Y;
    int C_X = B_X;

    int n_els_a = A_Y * A_X;
    int n_els_b = B_Y * B_X;
    int n_els_c = C_Y * C_X;

    // Type sizes
    size_t n_uint = sizeof(cl_uint);
    size_t n_ulong = sizeof(cl_ulong);
    size_t n_mem = sizeof(cl_mem);

    tensor *a = tensor_init_2d(A_Y, A_X);
    tensor *b = tensor_init_2d(B_Y, B_X);
    tensor *c = tensor_init_2d(C_Y, C_X);

    printf("** Matrix dimensions **\n");
    printf("  %-10s %6d %6d\n", "a", A_Y, A_X);
    printf("  %-10s %6d %6d\n", "b", B_Y, B_X);
    printf("  %-10s %6d %6d\n", "c", C_Y, C_X);
    printf("\n");
    printf("** Kernel setup **\n");
    printf("  %-10s %4d\n", "V size", v_size);
    printf("  %-10s %4d\n", "X scale", x_scale);
    printf("  %-10s %4d %4d\n", "PE dims", pe_s, pe_s);
    printf("  %-10s %4d %4d\n", "Block A", A_BLOCK_Y, A_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Block B", B_BLOCK_Y, B_BLOCK_X);
    printf("  %-10s %4d %4d\n", "Block C", C_BLOCK_Y, C_BLOCK_X);
    printf("\n");

    printf("** Initializing input matrices **\n");
    matrix_init(a, v_type);
    matrix_init(b, v_type);

    printf("** Multiplying on CPU **\n");
    tensor *c_ref = tensor_multiply_new(a, b);

    // Can we do this in one step?
    tensor *c_ref_tiled = tensor_tile_2d_new(
        c_ref, A_BLOCK_Y, B_BLOCK_X, 0, 0
    );
    int n_tiles = N * K * A_BLOCK_Y;
    tensor_set_dims(c_ref_tiled, 3, (int[]){n_tiles, pe_s, pe_s});


    tensor *c_ref_tiled_transposed = tensor_init_3d(n_tiles, pe_s, pe_s);
    tensor_transpose_tiled(c_ref_tiled, c_ref_tiled_transposed);
    tensor_set_dims(c_ref_tiled_transposed, 2, (int[]){C_Y, C_X});

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
        "-cl-std=CL2.0 "
        "-Werror "
        "-D PE_S=%d "
        "-D V_TYPE=%d "
        "-D V_SIZE=%d "
        "-D X_SCALE=%d ",
        pe_s, v_type, v_size, x_scale
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

    int size = V_TYPE_SIZE[v_type];
    size_t n_bytes_a = n_els_a * size;
    size_t n_bytes_b = n_els_b * size;
    size_t n_bytes_c = n_els_c * size;

    ocl_ctx_buf bufs[3] = {
        {0, n_bytes_a, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA},
        {0, n_bytes_b, CL_MEM_READ_ONLY | CL_CHANNEL_2_INTELFPGA},
        {0, n_bytes_c, CL_MEM_WRITE_ONLY | CL_CHANNEL_3_INTELFPGA}
    };
    for (int i = 0; i < 3; i++) {
        OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, bufs[i]));
    }

    // Extra buffers for casts
    void *dev_buf_a = malloc(n_bytes_a);
    void *dev_buf_b = malloc(n_bytes_b);
    void *dev_buf_c = malloc(n_bytes_c);

    buffer_copy_cast(
        dev_buf_a, v_type,
        a_tiled->data, V_TYPE_FLOAT,
        n_els_a
    );
    buffer_copy_cast(
        dev_buf_b, v_type,
        b_transpose_tiled->data, V_TYPE_FLOAT,
        n_els_b
    );

    printf("** Writing buffers **\n");
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
        ctx, 0, BUF_A, dev_buf_a
    ));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(
        ctx, 0, BUF_B, dev_buf_b
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
    printf("** Running kernel **\n");
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
    OCL_CHECK_ERR(ocl_ctx_read_buffer(ctx, 0, BUF_C, dev_buf_c));

    buffer_copy_cast(
        c->data, V_TYPE_FLOAT,
        dev_buf_c, v_type,
        n_els_c
    );
    //printf("Expected:\n");
    //tensor_print(c_ref_tiled_transposed, true, 2, 160, " ");
    if (v_type == V_TYPE_CHAR) {
        // Simulate wrap-around arithmetic of low-precision ints.
        tensor *t = c_ref_tiled_transposed;
        tensor_unary(t, t, TENSOR_UNARY_OP_REMAINDER, 256);
        for (int i = 0; i < tensor_n_elements(t); i++) {
            float v = t->data[i];
            if (v >= 128) {
                v -= 256;
            }
            t->data[i] = v;
        }
    }
    tensor_check_equal_contents(c, c_ref_tiled_transposed, 0.1);

    ocl_ctx_free(ctx);

    free(dev_buf_a);
    free(dev_buf_b);
    free(dev_buf_c);

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
