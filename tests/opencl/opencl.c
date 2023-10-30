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
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    assert(M % TILE_SIZE_SIMD == 0);
    assert(N % TILE_SIZE_SIMD == 0);
    assert(K % TILE_SIZE_SIMD == 0);

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
    OCL_CHECK_ERR(ocl_basic_setup(0, 0, &platform, &dev, &ctx, 1, &queue));
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
    OCL_CHECK_ERR(ocl_load_kernels(
                      ctx, dev, "libraries/opencl/matmul.cl",
                      3, kernel_names,
                      &program, kernels));

    // Create buffers
    cl_mem mem_a, mem_b, mem_c;
    OCL_CHECK_ERR(
        ocl_create_and_write_buffer(ctx, CL_MEM_READ_ONLY,
                                    queue, a->data,
                                    a_size, &mem_a
        ));
    OCL_CHECK_ERR(
        ocl_create_and_write_buffer(ctx, CL_MEM_READ_ONLY,
                                    queue, b->data,
                                    b_size, &mem_b
        ));
    OCL_CHECK_ERR(
        ocl_create_empty_buffer(ctx, CL_MEM_WRITE_ONLY,
                                c_size, &mem_c
        ));

    // Run kernels
    size_t local_sizes[3][2] = {
        // Sizes tuned for Quadro P400 card
        {TILE_SIZE_SIMD, TILE_SIZE_SIMD / WPT},
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
    OCL_CHECK_ERR(ocl_basic_setup(0, 0, &platform, &dev, &ctx, 1, &queue));
    ocl_print_device_details(dev, 0);
    printf("\n");

    uint32_t R = 100;
    uint32_t n_arr = 100 * 1000 * 1000;

    // Largest allocation on my GPU is 512mb
    uint32_t n_chunk = 50 * 1000 * 1000;

    size_t n_bytes_chunk = MIN(n_chunk, n_arr) * sizeof(cl_int);
    size_t n_bytes_arr = n_arr * sizeof(int32_t);

    printf("Allocating %ld bytes\n", n_bytes_arr);
    int32_t *arr = malloc(n_bytes_arr);
    rnd_pcg32_rand_range_fill((uint32_t *)arr, R + 1, n_arr);
    for (uint32_t i = 0; i < n_arr; i++) {
        arr[i] -= (R / 2);
    }

    int32_t sum = 0;
    uint64_t start = nano_count();
    for (uint32_t i = 0; i < n_arr; i++) {
        sum += arr[i];
    }
    printf("CPU sum %d\n", sum);
    printf("CPU sum took %.2fs\n", nanos_to_secs(nano_count() - start));


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
            ctx, dev, "tests/opencl/add_reduce.cl",
            1, (char *[]){"add_reduce"},
            &program, &kernel
        )
    );

    cl_ulong max_wg;
    OCL_CHECK_ERR(
        clGetDeviceInfo(
            dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(cl_ulong), &max_wg, NULL
        )
    );
    printf("Using workgroup size %lu\n", max_wg);

    uint32_t ofs = 0;
    int32_t ocl_sum = 0;
    start = nano_count();
    while (ofs < n_arr) {
        // Copy part to device
        OCL_CHECK_ERR(clEnqueueWriteBuffer(queue, mem_buf, CL_TRUE,
                                           0, n_bytes_chunk, &arr[ofs],
                                           0, NULL, NULL));
        // Run kernel
        OCL_CHECK_ERR(
            ocl_run_nd_kernel(
                queue, kernel, 1,
                (size_t[]){max_wg},
                (size_t[]){max_wg},
                3,
                sizeof(cl_uint), &n_chunk,
                sizeof(cl_mem), (void*)&mem_buf,
                sizeof(cl_mem), (void*)&mem_ret
            )
        );
        cl_int ret;
        OCL_CHECK_ERR(
            ocl_read_buffer(queue, &ret, sizeof(cl_int), mem_ret
            )
        );
        ocl_sum += ret;
        ofs += n_chunk;
    }
    printf("OpenCL sum %d\n", ocl_sum);
    printf("Took %.2fs\n", nanos_to_secs(nano_count() - start));

    free(arr);
    clReleaseMemObject(mem_buf);
    clReleaseMemObject(mem_ret);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(ctx);
}

// So many wrappers...
cl_int
ocl_ctx_add_tensor_buffer(ocl_ctx *me, cl_mem_flags flags, tensor *t) {
    size_t n_bytes = sizeof(float) * tensor_n_elements(t);
    return ocl_ctx_add_buffer(me, flags, n_bytes);
}

cl_int
ocl_ctx_write_tensor(ocl_ctx *me,
                     size_t queue_idx, size_t buffer_idx,
                     tensor *t) {
    size_t n_bytes = sizeof(float) * tensor_n_elements(t);
    return ocl_ctx_write_buffer(me, queue_idx, buffer_idx, t->data, n_bytes);
}

cl_int
ocl_ctx_read_tensor(ocl_ctx *me,
                    size_t queue_idx, size_t buffer_idx,
                    tensor *t) {
    size_t n_bytes = sizeof(float) * tensor_n_elements(t);
    return ocl_ctx_read_buffer(me, queue_idx, buffer_idx, t->data, n_bytes);
}


void
test_prefix_sum() {
    ocl_ctx *ctx = ocl_ctx_init(0, 0, true);
    OCL_CHECK_ERR(ctx->err);
    OCL_CHECK_ERR(
        ocl_ctx_load_kernels(
            ctx,
            "tests/opencl/prefix_sum.cl",
            1, (char *[]){"prefix_sum"}
        )
    );
    OCL_CHECK_ERR(ocl_ctx_add_queue(ctx));

    size_t n_arr = 10 * 1000 * 1000;
    tensor *arr = tensor_init(1, (int[]){n_arr});
    tensor *pf = tensor_init(1, (int[]){n_arr});
    tensor *pf_exp = tensor_init(1, (int[]){n_arr});
    tensor_fill_rand_range(arr, 2);
    tensor_unary(arr, arr, TENSOR_UNARY_OP_ADD, -1.0);

    // Get result from OpenCL
    OCL_CHECK_ERR(ocl_ctx_add_tensor_buffer(ctx, CL_MEM_READ_ONLY, arr));
    OCL_CHECK_ERR(ocl_ctx_add_tensor_buffer(ctx, CL_MEM_WRITE_ONLY, pf));
    OCL_CHECK_ERR(ocl_ctx_write_tensor(ctx, 0, 0, arr));
    PRINT_CODE_TIME(
        OCL_CHECK_ERR(
            ocl_ctx_run_kernel(
                ctx, 0, 0, 1, (size_t[]){128}, NULL,
                3,
                sizeof(cl_ulong), &n_arr,
                sizeof(cl_mem), (void *)&ctx->buffers[0],
                sizeof(cl_mem), (void *)&ctx->buffers[1]
            )
        ), "Took %.2fs\n"
    );
    OCL_CHECK_ERR(ocl_ctx_read_tensor(ctx, 0, 1, pf));

    // From CPU
    tensor_scan(arr, pf_exp, TENSOR_BINARY_OP_ADD, true, 0.0);

    // Numerical instabilities
    assert(tensor_check_equal(pf, pf_exp, 100));
    ocl_ctx_free(ctx);
    tensor_free(arr);
    tensor_free(pf);
    tensor_free(pf_exp);
}

void
test_count() {
    uint32_t n_els = 500 * 1000 * 1000;
    uint32_t n_bytes = sizeof(int32_t) * n_els;
    int32_t *arr = malloc_aligned(64, n_bytes);
    rnd_pcg32_rand_range_fill((uint32_t *)arr, 100, n_els);

    ocl_ctx *ctx = ocl_ctx_init(0, 0, true);
    OCL_CHECK_ERR(ocl_ctx_add_queue(ctx));

    OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, CL_MEM_READ_ONLY, n_bytes));
    OCL_CHECK_ERR(ocl_ctx_add_buffer(
                      ctx, CL_MEM_WRITE_ONLY, sizeof(int32_t)));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(ctx, 0, 0, arr, n_bytes));

    int32_t cnt[3];
    char *names[] = {
        "count_divisible",
        "count_divisible_simd",
        "count_divisible_simd2"
    };
    OCL_CHECK_ERR(ocl_ctx_load_kernels(
                      ctx, "tests/opencl/count.cl",
                      3, names));
    for (uint32_t i = 0; i < 3; i++) {
        char buf[256];
        sprintf(buf, "%-23s took %%.2fs\n", names[i]);
        PRINT_CODE_TIME(
            OCL_CHECK_ERR(ocl_ctx_run_kernel(
                              ctx, 0, i, 1,
                              (size_t[]){256}, NULL, 3,
                              sizeof(int32_t), &n_els,
                              sizeof(cl_mem), &ctx->buffers[0],
                              sizeof(cl_mem), &ctx->buffers[1])),
            buf);
        OCL_CHECK_ERR(ocl_ctx_read_buffer(
                          ctx, 0, 1, &cnt[i], sizeof(int32_t)));
    }
    assert(cnt[0] == cnt[1]);
    assert(cnt[1] == cnt[2]);

    free(arr);
    ocl_ctx_free(ctx);
}

typedef struct {
    uint32_t prio;
    uint32_t recv;
    uint32_t data;
    uint32_t padding;
} mail;

#define N_WORK_ITEMS     8

void
test_heap() {
    uint32_t N = 100;
    uint32_t T = 100;
    uint32_t n_bytes = sizeof(mail) * N * T;

    mail *msgs = malloc_aligned(64, n_bytes);
    for (uint32_t i = 0; i < N * T; i++) {
        msgs[i].prio = 1 + rnd_pcg32_rand_range(20);
        msgs[i].recv = rnd_pcg32_rand_range(N_WORK_ITEMS);
        msgs[i].data = rnd_pcg32_rand_range(10000);
    }

    /* uint32_t *arr = malloc_aligned(64, n_bytes); */
    /* rnd_pcg32_rand_range_fill(arr, 100, N * T); */

    ocl_ctx *ctx = ocl_ctx_init(0, 0, true);
    OCL_CHECK_ERR(ocl_ctx_add_queue(ctx));

    OCL_CHECK_ERR(ocl_ctx_load_kernels(
                      ctx, "tests/opencl/heap.cl",
                      1, (char *[]){"run_heap"}));

    OCL_CHECK_ERR(ocl_ctx_add_buffer(ctx, CL_MEM_READ_ONLY, n_bytes));
    OCL_CHECK_ERR(ocl_ctx_write_buffer(ctx, 0, 0, msgs, n_bytes));

    OCL_CHECK_ERR(ocl_ctx_run_kernel(
                      ctx, 0, 0, 1,
                      (size_t[]){N_WORK_ITEMS}, NULL, 3,
                      sizeof(uint32_t), &T,
                      sizeof(uint32_t), &N,
                      sizeof(cl_mem), &ctx->buffers[0]));

    ocl_ctx_free(ctx);
    free(msgs);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_add_reduce);
    PRINT_RUN(test_matmul);
    PRINT_RUN(test_prefix_sum);
    PRINT_RUN(test_count);
    PRINT_RUN(test_heap);
}
