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
    const int IMAGE_WIDTH = 1024;
    const int IMAGE_HEIGHT = 1024;
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

    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_int err = ocl_basic_setup(idx, 0,
                                 &platform, &device, &ctx);
    ocl_check_err(err);

    ocl_print_device_details(device, 0);

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, device, 0, &err);
    ocl_check_err(err);

    cl_program program;
    char *names[2] = {"dct8x8", "dct8x8_sd"};
    cl_kernel kernels[2];
    printf("* Loading kernel\n");
    err = ocl_load_kernels(ctx, device, fname,
                           2, names,
                           &program, kernels);
    ocl_check_err(err);

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
    cl_mem mem_image;
    err = ocl_create_and_fill_buffer(ctx, CL_MEM_READ_ONLY,
                                     queue,
                                     IMAGE_N_BYTES, image->data,
                                     &mem_image);

    cl_mem mem_dct = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                    IMAGE_N_BYTES, NULL, &err);
    ocl_check_err(err);


    // Run kernels
    printf("* Running kernel %s\n", names[0]);
    uint64_t start = nano_count();
    uint32_t n_iter = 1000;
    for (uint32_t i = 0; i < n_iter; i++) {
        ocl_run_nd_kernel(queue, kernels[0],
                          2, (size_t[]){
                              IMAGE_HEIGHT_BLOCKS,
                              IMAGE_WIDTH
                          }, NULL,
                          4,
                          sizeof(cl_mem), (void *)&mem_image,
                          sizeof(cl_mem), (void *)&mem_dct,
                          sizeof(cl_uint), (void *)&IMAGE_HEIGHT,
                          sizeof(cl_uint), (void *)&IMAGE_WIDTH);
    }
    uint64_t end = nano_count();
    double ms_per_kernel = ((double)(end - start) / 1000 / 1000) / n_iter;
    printf("\\--> %.2f ms/kernel\n", ms_per_kernel);

    printf("* Running kernel %s\n", names[1]);
    start = nano_count();
    n_iter = 1000;
    for (uint32_t i = 0; i < n_iter; i++) {
        ocl_run_nd_kernel(queue, kernels[1],
                          2, (size_t[]){
                              IMAGE_HEIGHT_BLOCKS,
                              IMAGE_WIDTH_BLOCKS
                          }, NULL,
                          4,
                          sizeof(cl_mem), (void *)&mem_image,
                          sizeof(cl_mem), (void *)&mem_dct,
                          sizeof(cl_uint), (void *)&IMAGE_HEIGHT,
                          sizeof(cl_uint), (void *)&IMAGE_WIDTH);
    }
    end = nano_count();
    ms_per_kernel = ((double)(end - start) / 1000 / 1000) / n_iter;
    printf("\\--> %.2f ms/kernel\n", ms_per_kernel);

    // Read data
    printf("* Reading device data\n");
    err = clEnqueueReadBuffer(queue, mem_dct, CL_TRUE,
                              0, IMAGE_N_BYTES, output->data,
                              0, NULL, NULL);
    ocl_check_err(err);

    if (IMAGE_WIDTH < 100) {
        printf("* Input:\n");
        tensor_print(image, "%4.0f", false);
        printf("* Reference:\n");
        tensor_print(ref, "%4.0f", false);
        printf("* Output:\n");
        tensor_print(output, "%4.0f", false);
    }
    tensor_check_equal(output, ref, 0.01);

    // Free tensors
    tensor_free(image);
    tensor_free(ref);
    tensor_free(output);

    // Teardown OpenCL

    // Release queue
    clFlush(queue);
    clFinish(queue);
    clReleaseCommandQueue(queue);

    // Free OpenCL memory
    clReleaseMemObject(mem_image);
    clReleaseMemObject(mem_dct);

    for (int i = 0; i < 2; i++) {
        clReleaseKernel(kernels[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(ctx);
    return 0;
}
