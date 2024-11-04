// Copyright (C) 2022,2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to use OpenCL pipes.
#include <assert.h>
#include <stdio.h>
#include "opencl/opencl.h"

int
main(int argc, char *argv[]) {

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
    OCL_CHECK_ERR(err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, dev, 0, &err);
    OCL_CHECK_ERR(err);

    // Load kernel
    cl_program program;
    cl_kernel kernel;
    printf("* Loading kernel\n");
    char *kernel_name = "producer";
    OCL_CHECK_ERR(ocl_load_kernels(
                      ctx, dev,
                      "programs/opencl/pipes.cl", "-cl-std=CL2.0 -Werror",
                      1, &kernel_name,
                      &program, &kernel
                  ));

    // Release queue
    OCL_CHECK_ERR(clFlush(queue));
    OCL_CHECK_ERR(clFinish(queue));
    OCL_CHECK_ERR(clReleaseCommandQueue(queue));

    OCL_CHECK_ERR(clReleaseKernel(kernel));
    OCL_CHECK_ERR(clReleaseProgram(program));
    OCL_CHECK_ERR(clReleaseContext(ctx));
    free(platforms);
    free(devices);

    return 0;
}
