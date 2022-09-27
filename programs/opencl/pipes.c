// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
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
    printf("* Loading kernel\n");
    assert(ocl_load_kernel(ctx, dev, "programs/opencl/pipes.cl",
                           &program, &kernel));

    // Release queue
    clFlush(queue);
    clFinish(queue);
    clReleaseCommandQueue(queue);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(ctx);
    free(platforms);
    free(devices);

    return 0;
}
