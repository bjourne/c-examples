// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"
#include "opencl/opencl.h"

void
test_load_kernel() {
    // Create tensor matrices.
    int SIZE = 1024;

    int a_rows = SIZE;
    int a_cols = SIZE;
    int b_rows = SIZE;
    int b_cols = SIZE;

    tensor *a = tensor_init(2, (int[]){a_rows, a_cols});
    tensor *b = tensor_init(2, (int[]){b_rows, b_cols});
    tensor *c = tensor_init(2, (int[]){a_rows, b_cols});
    tensor *c_exp = tensor_init(2, (int[]){a_rows, b_cols});

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
    ocl_check_err(ocl_load_kernels(ctx, dev, "libraries/opencl/matmul.cl",
                                   1, (char *[]){"matmul"},
                                   &program, &kernel));

    // Create buffers
    cl_mem mem_a, mem_b, mem_c;
    ocl_check_err(ocl_create_and_fill_buffer(ctx, CL_MEM_READ_ONLY,
                                             queue, a->data,
                                             a_size, &mem_a));
    ocl_check_err(ocl_create_and_fill_buffer(ctx, CL_MEM_READ_ONLY,
                                             queue, b->data,
                                             b_size, &mem_b));
    ocl_check_err(ocl_create_empty_buffer(ctx, CL_MEM_WRITE_ONLY,
                                          c_size, &mem_c));


    // Run kernel
    const size_t local[2] = {4, 4};
    const size_t global[2] = {a_rows, b_cols};

    err = ocl_run_nd_kernel(queue, kernel, 2, global, local, 6,
                            sizeof(int), (void*)&a_cols,
                            sizeof(int), (void*)&b_rows,
                            sizeof(int), (void*)&b_cols,
                            sizeof(cl_mem), (void*)&mem_a,
                            sizeof(cl_mem), (void*)&mem_b,
                            sizeof(cl_mem), (void*)&mem_c);

    // Multiply to reference
    tensor_multiply(a, b, c_exp);

    // Read from device
    err = clEnqueueReadBuffer(queue, mem_c, CL_TRUE,
                              0, c_size, c->data,
                              0, NULL, NULL);
    ocl_check_err(err);
    assert(tensor_check_equal(c_exp, c, 0.0001));

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
    tensor_free(c_exp);
}

int
main(int argc, char *argv[]) {
    rand_init(1234);
    PRINT_RUN(test_load_kernel);
}
