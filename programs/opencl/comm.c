// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how two kernels can communicate with each other.
#include <assert.h>
#include <stdio.h>
#include "opencl/opencl.h"

int
main(int argc, char *argv[]) {
    size_t n_cl_uint = sizeof(cl_uint);
    size_t n_cl_mem = sizeof(cl_mem);

    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    OCL_CHECK_ERR(
        ocl_basic_setup(
            1, 0,
            &platform, &device, &ctx, 0, NULL
        )
    );

    cl_int err;
    cl_command_queue_properties props[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, device, props, &err);
    OCL_CHECK_ERR(err);

    cl_uint zero = 0;
    cl_mem mem_buf;
    OCL_CHECK_ERR(
        ocl_create_and_fill_buffer(
            ctx, CL_MEM_READ_WRITE, queue,
            &zero, n_cl_uint, 2 * n_cl_uint, &mem_buf
        )
    );

    // Load program
    cl_program program;
    cl_kernel kernels[2];
    OCL_CHECK_ERR(
        ocl_load_kernels(
            ctx, device,
            "programs/opencl/comm.cl",
            2, (char*[]){"loop", "post"},
            &program, kernels
        )
    );

    // Run rogram
    cl_event events[2];
    OCL_CHECK_ERR(
        ocl_set_kernel_arguments(
            kernels[0], 1,
            n_cl_mem, (void *)&mem_buf
        )
    );
    OCL_CHECK_ERR(
        clEnqueueNDRangeKernel(
            queue, kernels[0], 1, NULL,
            (size_t[]){1}, NULL, 0, NULL,
            &events[0]
        )
    );

    OCL_CHECK_ERR(
        ocl_set_kernel_arguments(
            kernels[1], 1,
            n_cl_mem, (void *)&mem_buf
        )
    );
    OCL_CHECK_ERR(
        clEnqueueNDRangeKernel(
            queue, kernels[1], 1, NULL,
            (size_t[]){1}, NULL, 0, NULL,
            &events[1]
        )

    );
    OCL_CHECK_ERR(clWaitForEvents(2, events));

    cl_uint ret[2] = {0};
    OCL_CHECK_ERR(ocl_read_buffer(queue, ret, 2 * sizeof(cl_uint), mem_buf));
    printf("got %u, %u\n", ret[0], ret[1]);

    // Release OpenCL resources
    clFlush(queue);
    clFinish(queue);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(mem_buf);
    for (uint32_t i = 0; i < 2; i++) {
        clReleaseKernel(kernels[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(ctx);
    return 0;
}
