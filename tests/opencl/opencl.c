// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "opencl/opencl.h"
#include "opencl/matmul.h"
#include "random/random.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"

static void
init_arrays(tensor *a, tensor *b) {
    tensor_fill_rand_range(a, 100);
    tensor_fill_rand_range(b, 100);
}

void
test_matmul() {
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
        OCL_CHECK_ERR(ocl_run_nd_kernel(
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

void
test_add_reduce() {
    cl_platform_id platform;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue queue;
    ocl_check_err(ocl_basic_setup(1, 0, &platform, &dev, &ctx, 1, &queue));
    ocl_print_device_details(dev, 0);
    printf("\n");

    uint32_t R = 100;
    uint32_t n_arr = 2 * 1024 * 1000 * 1000;

    // Largest allocation on my GPU is 512mb
    uint32_t n_chunk = 128 * 1000 * 1000;

    size_t n_bytes_chunk = n_chunk * sizeof(cl_int);
    size_t n_bytes_arr = n_arr * sizeof(int32_t);

    int32_t *arr = malloc(n_bytes_arr);
    rnd_pcg32_rand_range_fill((uint32_t *)arr, R + 1, n_arr);
    for (uint32_t i = 0; i < n_arr; i++) {
        arr[i] -= (R / 2);
    }

    int32_t sum = 0;
    PRINT_CODE_TIME({
            for (uint32_t i = 0; i < n_arr; i++) {
                sum += arr[i];
            }
        }, "CPU sum took %.2fs\n");
    printf("CPU sum %d\n", sum);

    // Create OpenCL buffers
    cl_mem mem_buf, mem_ret;
    OCL_CHECK_ERR(
        ocl_create_empty_buffer(
            ctx, CL_MEM_READ_ONLY,
            n_bytes_chunk,
            &mem_buf
        )
    );
    OCL_CHECK_ERR(
        ocl_create_empty_buffer(
            ctx, CL_MEM_READ_ONLY,
            sizeof(cl_int),
            &mem_ret
        )
    );

    cl_program program;
    cl_kernel kernel;
    OCL_CHECK_ERR(
        ocl_load_kernels(
            ctx, dev, "libraries/opencl/add_reduce.cl",
            1, (char *[]){"add_reduce"},
            &program, &kernel
        )
    );

    uint32_t ofs = 0;
    int32_t ocl_sum = 0;
    while (ofs < n_arr) {
        // Copy part to device
        OCL_CHECK_ERR(clEnqueueWriteBuffer(queue, mem_buf, CL_TRUE,
                                           0, n_bytes_chunk, &arr[ofs],
                                           0, NULL, NULL));
        // Run kernel
        OCL_CHECK_ERR(
            ocl_run_nd_kernel(
                queue, kernel, 1,
                (size_t[]){512},
                (size_t[]){512},
                3,
                sizeof(cl_uint), &n_chunk,
                sizeof(cl_mem), (void*)&mem_buf,
                sizeof(cl_mem), (void*)&mem_ret
            )
        );
        cl_int ret;
        OCL_CHECK_ERR(ocl_read_buffer(queue, &ret, sizeof(cl_int), mem_ret));
        ocl_sum += ret;
        ofs += n_chunk;
    }
    printf("OpenCL sum %d\n", ocl_sum);

    free(arr);
    clReleaseMemObject(mem_buf);
    clReleaseMemObject(mem_ret);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(ctx);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_add_reduce);
    PRINT_RUN(test_matmul);
}
