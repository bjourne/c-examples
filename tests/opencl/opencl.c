// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"
#include "opencl/opencl.h"

void
test_load_kernel() {
    // Create tensor matrices.
    int SIZE = 2048;

    int a_rows = SIZE;
    int a_cols = SIZE;
    int b_rows = SIZE;
    int b_cols = SIZE;

    tensor *a = tensor_init(2, (int[]){a_rows, a_cols});
    tensor *b = tensor_init(2, (int[]){b_rows, b_cols});
    tensor *c = tensor_init(2, (int[]){a_rows, b_cols});

    int a_els = a_rows * a_cols;
    int b_els = b_rows * b_cols;
    int c_els = a_rows * b_cols;

    size_t a_size = a_els * sizeof(float);
    size_t b_size = b_els * sizeof(float);
    size_t c_size = c_els * sizeof(float);

    tensor_fill_rand_range(a, 100);
    tensor_fill_rand_range(b, 100);

    cl_int err;
    cl_uint n_platforms;
    cl_platform_id *platforms;
    ocl_get_platforms(&n_platforms, &platforms);

    cl_uint n_devices;
    cl_device_id *devices;
    ocl_get_devices(platforms[0], &n_devices, &devices);

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
    assert(ocl_load_kernels(ctx, dev, "libraries/opencl/matmul.cl",
                            1, (char *[]){"matmul"},
                            &program, &kernel));

    // Create buffers
    cl_mem mem_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY, a_size, NULL, NULL);
    cl_mem mem_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, b_size, NULL, NULL);
    cl_mem mem_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, c_size, NULL, NULL);

    // Queue buffers
    clEnqueueWriteBuffer(queue, mem_a, CL_TRUE, 0, a_size, a->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, mem_b, CL_TRUE, 0, b_size, b->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, mem_c, CL_TRUE, 0, c_size, c->data, 0, NULL, NULL);

    // Kernel args
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&a_cols);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&b_rows);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&b_cols);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mem_a);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mem_b);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mem_c);

    // Thread sizes
    const size_t local[2] = {32, 32};
    const size_t global[2] = {a_rows, b_cols};
    cl_event event = NULL;

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
                                 global, local, 0, NULL, &event);
    ocl_check_err(err);

    err = clWaitForEvents(1, &event);
    ocl_check_err(err);

    // Release queue
    clFlush(queue);
    clFinish(queue);
    clReleaseCommandQueue(queue);

    // Freeing buffers
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(ctx);
    free(platforms);
    free(devices);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

int
main(int argc, char *argv[]) {
    rand_init(1234);
    PRINT_RUN(test_load_kernel);
}
