// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"
#include "opencl/opencl.h"
#include "opencl/matmul.h"

static void
init_arrays(tensor *a, tensor *b) {
    tensor_fill_rand_range(a, 100);
    tensor_fill_rand_range(b, 100);
    /* for (int i = 0; i < a->dims[0]; i++) { */
    /*     for (int j = 0; j < a->dims[1]; j++) { */
    /*         a->data[i * a->dims[1] + j] = i; */
    /*     } */
    /* } */
    /* for (int i = 0; i < b->dims[0]; i++) { */
    /*     for (int j = 0; j < b->dims[1]; j++) { */
    /*         b->data[i * b->dims[1] + j] = j; */
    /*     } */
    /* } */
    /* printf("== A ==\n"); */
    /* tensor_print(a, "%5.0f", false); */
    /* printf("== B ==\n"); */
    /* tensor_print(b, "%5.0f", false); */
}

void
test_load_kernel() {
    // A has dimension MxK, b KxN, and c MxN
    const int M = 4096;
    const int N = 2048;
    const int K = 4096;

    assert(M % TILE_SIZE == 0);
    assert(N % TILE_SIZE == 0);
    assert(K % TILE_SIZE == 0);

    tensor *a = tensor_init(2, (int[]){M, K});
    tensor *b = tensor_init(2, (int[]){K, N});
    tensor *c = tensor_init(2, (int[]){M, N});
    tensor *c_exp = tensor_init(2, (int[]){M, N});

    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    init_arrays(a, b);

    // Multiply to reference
    tensor_multiply(a, b, c_exp);

    cl_platform_id platform;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue queue;
    ocl_check_err(ocl_basic_setup(0, 0, &platform, &dev, &ctx, 1, &queue));
    ocl_print_device_details(dev, 0);
    printf("\n");

    // Load kernels
    char *kernel_names[] = {
        "matmul_tiled_simd",
        "matmul_tiled",
        "matmul_naive"
    };
    cl_program program;
    cl_kernel kernels[3];
    ocl_check_err(ocl_load_kernels(
                      ctx, dev, "libraries/opencl/matmul.cl",
                      3, kernel_names,
                      &program, kernels));

    // Create buffers
    cl_mem mem_a, mem_b, mem_c;
    ocl_check_err(ocl_create_and_write_buffer(ctx, CL_MEM_READ_ONLY,
                                             queue, a->data,
                                             a_size, &mem_a));
    ocl_check_err(ocl_create_and_write_buffer(ctx, CL_MEM_READ_ONLY,
                                              queue, b->data,
                                              b_size, &mem_b));
    ocl_check_err(ocl_create_empty_buffer(ctx, CL_MEM_WRITE_ONLY,
                                          c_size, &mem_c));

    // Run kernels
    size_t local_sizes[3][2] = {
        // Sizes tuned for Quadro P400 card
        {TILE_SIZE, TILE_SIZE / WPT},
        {TILE_SIZE, TILE_SIZE},
        {8, 8}
    };
    size_t global_sizes[3][2] = {
        {M, N / WPT},
        {M, N},
        {M, N}
    };

    for (int i = 0; i < 3; i++) {
        uint64_t start = nano_count();
        ocl_check_err(ocl_run_nd_kernel(
                          queue, kernels[i], 2,
                          global_sizes[i],
                          local_sizes[i],
                          6,
                          sizeof(int), (void*)&M,
                          sizeof(int), (void*)&N,
                          sizeof(int), (void*)&K,
                          sizeof(cl_mem), (void*)&mem_a,
                          sizeof(cl_mem), (void*)&mem_b,
                          sizeof(cl_mem), (void*)&mem_c));
        printf("Kernel %-20s: %6.2fs\n",
               kernel_names[i],
               nanos_to_secs(nano_count() - start));

        // Read from device
        ocl_check_err(ocl_read_buffer(queue, c->data, c_size, mem_c));

        /* printf("== C ==\n"); */
        /* tensor_print(c, "%5.0f", false); */

        /* printf("== C ref ==\n"); */
        /* tensor_print(c_exp, "%5.0f", false); */

        assert(tensor_check_equal(c_exp, c, 0.0001));
    }

    // Release queue
    clFlush(queue);
    clFinish(queue);
    clReleaseCommandQueue(queue);

    // Freeing buffers
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);

    for (size_t i = 0; i < ARRAY_SIZE(kernels); i++) {
        clReleaseKernel(kernels[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(ctx);

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
