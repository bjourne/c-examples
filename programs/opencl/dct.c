// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Benchmarks 8x8 dct on tensors
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/common.h"
#include "opencl/opencl.h"
#include "tensors/tensors.h"
#include "tensors/dct.h"

// Should match constants in dct8x8.cl
#define BLOCK_SIZE 8

int
main(int argc, char *argv[]) {
    const int IMAGE_WIDTH = 16;
    const int IMAGE_HEIGHT = 16;
    const int IMAGE_WIDTH_BLOCKS = IMAGE_WIDTH / BLOCK_SIZE;
    const int IMAGE_HEIGHT_BLOCKS = IMAGE_HEIGHT / BLOCK_SIZE;
    const int IMAGE_N_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);

    // Load kernel
    float x[8] = {20, 9, 10, 11, 12, 13, 14, 15};
    float y[8], y2[8];
    tensor_dct8_nvidia(x, y);
    tensor_dct8_loeffler(x, y2);
    for (int i = 0; i < 8; i++) {
        printf("%2d %5.2f %5.2f\n", i, y[i], y2[i]);
    }
    printf("\n");

    // Setup OpenCL
    if (argc != 3) {
        printf("Usage: %s platform-id kernel-path\n", argv[0]);
        printf("E.g. programs/opencl/dct8x8.cl for kernel path.\n");
        exit(1);
    }

    int idx = atoi(argv[1]);
    char *fname = argv[2];

    ocl_ctx *ctx = ocl_ctx_init(idx, 0, true);
    OCL_CHECK_ERR(ctx->err);
    OCL_CHECK_ERR(ocl_ctx_add_queue(ctx));

    //cl_program program;
    char *names[2] = {"dct8x8", "dct8x8_sd"};
    //cl_kernel kernels[2];
    printf("* Loading kernels\n");
    OCL_CHECK_ERR(ocl_ctx_load_kernels(
                      ctx, "-cl-std=CL2.0 -Werror",
                      fname,
                      2, names));

    // Allocate and initialize tensors
    printf("* Initializing tensors\n");
    int dims[] = {IMAGE_HEIGHT, IMAGE_WIDTH};
    tensor *image = tensor_init(2, dims);
    tensor *ref = tensor_init(2, dims);
    tensor *output = tensor_init(2, dims);
    tensor_fill_rand_range(image, 100);

    // Compute reference results
    tensor_dct2d_8x8_blocks(image, ref, true);

    // One buffer for the input to the kernel and another one for the
    // dct-ized result.
    OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, CL_MEM_READ_ONLY, IMAGE_N_BYTES));
    OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, CL_MEM_WRITE_ONLY, IMAGE_N_BYTES));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(ctx, 0, 0, image->data, IMAGE_N_BYTES));

    size_t wg_sizes[2][2] = {
        {IMAGE_HEIGHT_BLOCKS, IMAGE_WIDTH_BLOCKS},
        {IMAGE_HEIGHT_BLOCKS, IMAGE_WIDTH_BLOCKS}
    };

    for (uint32_t i = 1; i < 2; i++) {
        printf("* Running kernel %s\n", names[i]);
        uint64_t start = nano_count();
        uint32_t n_iter = 10;
        for (uint32_t j = 0; j < n_iter; j++) {
            OCL_CHECK_ERR(ocl_ctx_run_kernel(
                              ctx, 0, i,
                              2, wg_sizes[i], NULL,
                              4,
                              sizeof(cl_mem), (void *)&ctx->buffers[0],
                              sizeof(cl_mem), (void *)&ctx->buffers[1],
                              sizeof(cl_uint), (void *)&IMAGE_HEIGHT,
                              sizeof(cl_uint), (void *)&IMAGE_WIDTH));
        }
        uint64_t end = nano_count();
        double ms_per_kernel = ((double)(end - start) / 1000 / 1000) / n_iter;
        printf("\\--> %.2f ms/kernel\n", ms_per_kernel);

        OCL_CHECK_ERR(ocl_ctx_read_buffer(ctx, 0, 1,
                                          output->data, IMAGE_N_BYTES));
        tensor_check_equal(output, ref, 0.01);
    }

    if (IMAGE_WIDTH < 100) {
        printf("* Input:\n");
        tensor_print(image, true, 0, 80, " ");
        printf("* Reference:\n");
        tensor_print(ref, true, 0, 80, " ");
        printf("* Output:\n");
        tensor_print(output, true, 0, 80, " ");
    }

    // Free tensors
    tensor_free(image);
    tensor_free(ref);
    tensor_free(output);

    ocl_ctx_free(ctx);
    return 0;
}
