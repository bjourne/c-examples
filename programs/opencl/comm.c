// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how two kernels can communicate with each other using
// the cl_intel_channels extension.
#include <assert.h>
#include <stdio.h>
#include "opencl/opencl.h"

#define N_KERNELS    2

// This type may be useful.
typedef struct {
    cl_mem mem;
    void *ptr;
    size_t n_bytes;
} ocl_dblbuf;

cl_int
ocl_dblbuf_init(ocl_dblbuf *buf,
                cl_context ctx,
                cl_mem_flags flags,
                size_t n_bytes) {
    buf->n_bytes = n_bytes;
    buf->ptr = malloc(n_bytes);
    cl_int err;
    buf->mem = clCreateBuffer(ctx, flags, n_bytes, NULL, &err);
    return err;
}

void
ocl_dblbuf_free(ocl_dblbuf *buf) {
    clReleaseMemObject(buf->mem);
    free(buf->ptr);
}

cl_int
ocl_dblbuf_push(ocl_dblbuf *buf, cl_command_queue queue) {
    return clEnqueueWriteBuffer(
        queue, buf->mem, CL_TRUE,
        0, buf->n_bytes, buf->ptr,
        0, NULL, NULL);
}

cl_int
ocl_dblbuf_pull(ocl_dblbuf *buf, cl_command_queue queue) {
    return clEnqueueReadBuffer(
        queue, buf->mem, CL_TRUE,
        0, buf->n_bytes, buf->ptr,
        0, NULL, NULL);
}

int
main(int argc, char *argv[]) {
    // Setup OpenCL
    if (argc != 3) {
        printf("Usage: %s platform-id kernel-path\n", argv[0]);
        printf("E.g. programs/opencl/comm.cl for kernel path.\n");
        exit(1);
    }
    int plat_idx = atoi(argv[1]);

    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queues[N_KERNELS];
    OCL_CHECK_ERR(
        ocl_basic_setup(
            plat_idx, 0,
            &platform, &device, &ctx,
            N_KERNELS, queues
        )
    );
    ocl_print_device_details(device, NULL);

    // Setup IO memory
    const uint32_t N_ARR = 200;

    ocl_dblbuf arr, ret;
    OCL_CHECK_ERR(
        ocl_dblbuf_init(
            &arr, ctx,
            CL_MEM_READ_ONLY, sizeof(cl_int) * N_ARR
        )
    );
    OCL_CHECK_ERR(
        ocl_dblbuf_init(
            &ret, ctx,
            CL_MEM_WRITE_ONLY, sizeof(cl_int) * 1
        )
    );
    for (uint32_t i = 0; i < N_ARR; i++) {
        ((int32_t *)arr.ptr)[i] = i;
    }
    OCL_CHECK_ERR(ocl_dblbuf_push(&arr, queues[0]));

    // Load program
    cl_program program;
    cl_kernel kernels[2];
    OCL_CHECK_ERR(
        ocl_load_kernels(
            ctx, device,
            argv[2],
            2, (char*[]){"consumer", "producer"},
            &program, kernels
        )
    );

    // Launch
    cl_event events[2];
    OCL_CHECK_ERR(
        ocl_set_kernel_arguments(
            kernels[0], 2,
            sizeof(uint32_t), &N_ARR,
            sizeof(cl_mem), (void *)&ret.mem
        )
    );
    OCL_CHECK_ERR(
        ocl_set_kernel_arguments(
            kernels[1], 2,
            sizeof(uint32_t), &N_ARR,
            sizeof(cl_mem), (void *)&arr.mem
        )
    );
    OCL_CHECK_ERR(
        clEnqueueNDRangeKernel(queues[0], kernels[0], 1, NULL,
                               (size_t[]){1}, NULL, 0, NULL, &events[0]
        )
    );
    OCL_CHECK_ERR(
        clEnqueueNDRangeKernel(queues[1], kernels[1], 1, NULL,
                               (size_t[]){1}, NULL, 0, NULL, &events[1]
        )
    );
    clWaitForEvents(2, events);

    ocl_dblbuf_pull(&ret, queues[0]);
    printf("Answer: %d\n", ((int32_t *)ret.ptr)[0]);


    ocl_dblbuf_free(&arr);
    ocl_dblbuf_free(&ret);

    for (uint32_t i = 0; i < N_KERNELS; i++) {
        clReleaseCommandQueue(queues[i]);
        clReleaseKernel(kernels[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(ctx);
    return 0;
}
