// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Benchmarks 8x8 dct on tensors
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/common.h"
#include "opencl/opencl.h"
#include "tensors/tensors.h"
#include "tensors/tensors-dct.h"

int
main(int argc, char *argv[]) {

    const int IMAGE_WIDTH = 256;
    const int IMAGE_HEIGHT = 256;
    const int IMAGE_N_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);

    const int BLOCKDIM_X = 8;
    const int BLOCKDIM_Y = 8;
    const int BLOCK_SIZE = 8;

    //  Single floats
    const int SIMD_LOC = 1;

    float x[8] = {20, 9, 10, 11, 12, 13, 14, 15};
    float y[8], y2[8];
    tensor_dct8_nvidia(x, y);
    tensor_dct8_loeffler(x, y2);
    //loeffler8(x, y2);
    for (int i = 0; i < 8; i++) {
        printf("%2d %5.2f %5.2f\n", i, y[i], y2[i]);
    }
    printf("\n");

    // Setup OpenCL
    cl_int err;
    cl_uint n_platforms;
    cl_platform_id *platforms;
    ocl_get_platforms(&n_platforms, &platforms);

    cl_uint n_devices;
    cl_device_id *devices;
    ocl_get_devices(platforms[1], &n_devices, &devices);

    cl_device_id dev = devices[0];
    ocl_print_device_details(dev, 0);

    cl_context ctx = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
    ocl_check_err(err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, dev, 0, &err);
    ocl_check_err(err);

    // Load kernel
    cl_program program;
    cl_kernel kernel;
    printf("* Loading kernel\n");
    ocl_load_kernel(ctx, dev, "programs/opencl/dct8x8.cl",
                    &program, &kernel);

    // Allocate and initialize tensors
    printf("* Initializing tensors\n");
    int dims[] = {IMAGE_HEIGHT, IMAGE_WIDTH};
    tensor *image = tensor_init(2, dims);
    tensor *ref = tensor_init(2, dims);
    tensor *output = tensor_init(2, dims);
    tensor_randrange(image, 100);

    // Compute reference results
    tensor_dct2d_blocks(image, ref, 8, 8);

    // Allocate and write to OpenCL buffers
    cl_mem mem_image = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      IMAGE_N_BYTES, NULL, &err);
    ocl_check_err(err);
    cl_mem mem_dct = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                    IMAGE_N_BYTES, NULL, &err);
    ocl_check_err(err);
    err = clEnqueueWriteBuffer(queue, mem_image, CL_TRUE,
                               0, IMAGE_N_BYTES, image->data,
                               0, NULL, NULL);
    ocl_check_err(err);

    // Run kernel
    printf("* Running kernel\n");
    size_t local[] = {
        BLOCKDIM_Y / BLOCK_SIZE,
        BLOCKDIM_X / SIMD_LOC
    };
    size_t global[] = {
        IMAGE_HEIGHT / BLOCKDIM_Y * local[0],
        IMAGE_WIDTH / BLOCKDIM_X * local[1]
    };

    uint64_t start = nano_count();
    for (int i = 0; i < 1000; i++) {
        ocl_run_nd_kernel(queue, kernel,
                          2, global, local,
                          8,
                          sizeof(cl_mem), (void *)&mem_image,
                          sizeof(cl_mem), (void *)&mem_dct,
                          sizeof(cl_uint), (void *)&IMAGE_HEIGHT,
                          sizeof(cl_uint), (void *)&IMAGE_WIDTH);
    }

    uint64_t end = nano_count();
    double secs = (double)(end - start) / 1000 / 1000 / 1000;
    printf("\\--> %.3f seconds\n", secs);

    // Read data
    printf("* Reading device data\n");
    err = clEnqueueReadBuffer(queue, mem_dct, CL_TRUE,
                              0, IMAGE_N_BYTES, output->data,
                              0, NULL, NULL);
    ocl_check_err(err);

    /* printf("* Input:\n"); */
    /* tensor_print(image, "%4.0f", false); */
    /* printf("* Reference:\n"); */
    /* tensor_print(ref, "%4.0f", false); */
    /* printf("* Output:\n"); */
    /* tensor_print(output, "%4.0f", false); */

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

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(ctx);
    free(platforms);
    free(devices);

    return 0;
}
