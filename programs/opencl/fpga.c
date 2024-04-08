// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"
#include "files/files.h"
#include "linalg/linalg.h"
#include "opencl/opencl.h"
#include "tensors/tensors.h"
#include "tensors/multiply.h"

#define HOST
#include "matmul_fpga_config.h"

#define MAT_A_NUM_BLOCKS_IN_ROW             (WA / MAT_A_BLOCK_WIDTH)
#define MAT_A_NUM_BLOCKS_IN_COL             (HA / MAT_A_BLOCK_HEIGHT)
#define MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS  (MAT_A_NUM_BLOCKS_IN_ROW * MAT_A_BLOCK_NUM_VECTORS)

#define MAT_B_NUM_BLOCKS_IN_ROW             (WB / MAT_B_BLOCK_WIDTH)
#define MAT_B_NUM_BLOCKS_IN_COL             (HB / MAT_B_BLOCK_HEIGHT)
#define MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS  (MAT_B_NUM_BLOCKS_IN_COL * MAT_B_BLOCK_NUM_VECTORS)
#define MAT_B_NUM_VECTORS_IN_MATRIX         (MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS * MAT_B_NUM_BLOCKS_IN_ROW)

#define MAT_B_BLOCK_HEIGHT          MAT_A_BLOCK_WIDTH
#define MAT_B_BLOCK_WIDTH           (COLUMNS_INTERLEAVED * PE_COLS)

#define HA (4 * MAT_A_BLOCK_HEIGHT)             // Matrix A height
#define WA (8 * MAT_A_BLOCK_WIDTH)             // Matrix A width

#define HB WA                                   // Matrix B height
#define WB (4 * MAT_B_BLOCK_WIDTH)              // Matrix B width

#define HC HA                                   // Matrix C height
#define WC WB                                   // Matrix C width

// Gold stuff
#define COMPUTE_GOLD_BLOCK_SIZE 64

#define HA_trim (2 *  MAT_A_BLOCK_HEIGHT)
#define WB_trim WB

static  void
reorder_within_blocks(float * src,
                      float * dst,
                      int height, int width,
                      int num_sys_arr_columns, int block_width) {
    int num_elems = height * width;
    int column_interleaving = block_width / num_sys_arr_columns;
    int word_id = 0;
    for (int i = 0; i < num_elems; i += block_width) {
        for (int j = 0; j < column_interleaving; j++) {
            for (int k = 0; k < num_sys_arr_columns ; k++) {
                dst[word_id] = src[i + j + k * column_interleaving];
                word_id++;
            }
        }
    }
    assert(word_id == num_elems);
}

int
main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s platform-index kernel-path\n", argv[0]);
        exit(1);
    }

    tensor *a = tensor_init(2, (int[]){HA, WA});
    tensor *a_blocked = tensor_init(2, (int[]){HA, WA});
    tensor *b_transpose = tensor_init(2, (int[]){WB, HB});
    tensor *b_transpose_blocked = tensor_init(2, (int[]){WB, HB});
    tensor *b = tensor_init(2, (int[]){HB, WB});

    tensor *c = tensor_init(2, (int[]){HC, WC});
    tensor *c_golden = tensor_init(2, (int[]){HC, WC});
    tensor *c_golden_blocked = tensor_init(2, (int[]){HC, WC});
    tensor *c_golden_blocked_reordered = tensor_init(2, (int[]){HC, WC});
    tensor *c_ref = tensor_init(2, (int[]){HC, WC});

    printf("** Matrix dimensions **\n");
    printf("%12s %4d %4d\n", "a", HA, WA);
    printf("%12s %4d %4d\n", "b", HB, WB);
    printf("%12s %4d %4d\n", "c", HC, WC);
    printf("%12s %4d %4d\n", "a_blocked", HA, WA);
    printf("%12s %4d %4d\n", "b_transpose", WB, HB);
    printf("\n");
    printf("** Kernel setup **\n");
    printf("%12s %4d %4d\n", "PE dims", PE_ROWS, PE_COLS);
    printf("%12s %4d %4d\n", "Block A", MAT_A_BLOCK_HEIGHT, MAT_A_BLOCK_WIDTH);
    printf("%12s %4d %4d\n", "Block B", MAT_B_BLOCK_HEIGHT, MAT_B_BLOCK_WIDTH);
    printf("%12s %4d %4d\n", "Block C", MAT_C_BLOCK_HEIGHT, MAT_C_BLOCK_WIDTH);
    printf("%12s %4d %4d\n", "Interleave", ROWS_INTERLEAVED, COLUMNS_INTERLEAVED);
    printf("\n");

    printf("** Initializing input matrices **\n");
    tensor_fill_rand_range(a, 10);
    tensor_fill_rand_range(b, 10);
    tensor_fill_const(c, 0);

    printf("** Multiplying on CPU**\n");
    tensor_multiply(a, b, c_ref);

    tensor_linearize_tiles(a, a_blocked, MAT_A_BLOCK_HEIGHT, MAT_A_BLOCK_WIDTH);
    tensor_transpose(b, b_transpose);
    tensor_linearize_tiles(b_transpose, b_transpose_blocked,
                           MAT_B_BLOCK_WIDTH, MAT_B_BLOCK_HEIGHT);

    // Golden compute
    tensor_multiply(a, b, c_golden);
    tensor_linearize_tiles(c_golden, c_golden_blocked,
                           MAT_C_BLOCK_HEIGHT, MAT_C_BLOCK_WIDTH);
    reorder_within_blocks(c_golden_blocked->data,
                          c_golden_blocked_reordered->data,
                          HC, WC,
                          PE_COLS,
                          MAT_C_BLOCK_WIDTH);

    printf("** Setting up OpenCL **\n");

    cl_platform_id platform;
    cl_device_id dev;
    cl_context ctx;
    int plat_idx = atoi(argv[1]);
    OCL_CHECK_ERR(ocl_basic_setup(plat_idx, 0, &platform, &dev, &ctx, 0, NULL));
    ocl_print_device_details(dev, 0);

    printf("Loading kernels\n");
    cl_program program;
    cl_kernel kernels[3];
    OCL_CHECK_ERR(ocl_load_kernels(
                      ctx, dev,
                      argv[2], "-cl-std=CL2.0 -Werror",
                      3, (char *[]){"loadA", "loadB", "store"},
                      &program, kernels));

    // Create four queues
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    cl_command_queue queues[4];
    cl_int err;
    for (int i = 0; i < 4; i++) {
        queues[i] = clCreateCommandQueueWithProperties(ctx, dev,
                                                       props, &err);
        OCL_CHECK_ERR(err);
    }

    printf("Creating device buffers\n");
    size_t n_bytes_a = HA * WA * sizeof(float);
    size_t n_bytes_b = HB * WB * sizeof(float);
    size_t n_bytes_c = HC * WC * sizeof(float);

    cl_mem dev_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n_bytes_a,
                                  NULL, &err);
    OCL_CHECK_ERR(err);
    cl_mem dev_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n_bytes_b,
                                  NULL, &err);
    OCL_CHECK_ERR(err);
    cl_mem dev_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n_bytes_c,
                                  NULL, &err);
    OCL_CHECK_ERR(err);

    printf("Writing to device buffer A\n");
    err = clEnqueueWriteBuffer(queues[0], dev_a, CL_TRUE, 0, n_bytes_a,
                               a_blocked->data, 0, NULL, NULL);
    OCL_CHECK_ERR(err);

    printf("Writing to device buffer B\n");
    err = clEnqueueWriteBuffer(queues[1], dev_b, CL_TRUE, 0, n_bytes_a,
                               b_transpose_blocked->data, 0, NULL, NULL);
    OCL_CHECK_ERR(err);


    // LoadA kernel
    printf("Initializing LoadA kernel\n");
    unsigned int mat_a_num_vectors_in_row_of_blocks =
        MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS;
    unsigned char mat_a_num_blocks_in_col = MAT_A_NUM_BLOCKS_IN_COL;
    unsigned char mat_b_num_blocks_in_row = MAT_B_NUM_BLOCKS_IN_ROW;

    OCL_CHECK_ERR(ocl_set_kernel_arguments(
                      kernels[0], 4,
                      sizeof(cl_mem), (void *)&dev_a,
                      sizeof(unsigned int), (void *)&mat_a_num_vectors_in_row_of_blocks,
                      sizeof(unsigned char), (void *)&mat_a_num_blocks_in_col,
                      sizeof(unsigned char), (void *)&mat_b_num_blocks_in_row));

    // LoadB kernel
    printf("Initializing LoadB kernel\n");
    unsigned int mat_b_num_vectors_in_col_of_blocks =
        MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS;
    unsigned int mat_b_num_vectors_in_matrix =
        MAT_B_NUM_VECTORS_IN_MATRIX;

    ocl_set_kernel_arguments(
        kernels[1], 4,
        sizeof(cl_mem), (void *)&dev_b,
        sizeof(unsigned int), (void *)&mat_b_num_vectors_in_col_of_blocks,
        sizeof(unsigned int), (void *)&mat_b_num_vectors_in_matrix,
        sizeof(unsigned char), (void *)&mat_a_num_blocks_in_col);

    // Store kernel
    int mat_c_num_coalesced_words = WC * HC / PE_COLS;

    ocl_set_kernel_arguments(
        kernels[2], 2,
        sizeof(cl_mem), (void *)&dev_c,
        sizeof(int), (void *)&mat_c_num_coalesced_words);

    // Queue kernels
    size_t local[] = {1};
    size_t global[] = {1};
    cl_event events[3];
    for (int i = 0; i < 3; i++) {
        err = clEnqueueNDRangeKernel(queues[i], kernels[i],
                                     1, NULL, global, local,
                                     0, NULL,
                                     &events[i]);
        OCL_CHECK_ERR(err);
    }
    for(int i=0; i < 3; i++) {
        err = clFlush(queues[i]);
        OCL_CHECK_ERR(err);
    }

    for(int i = 0; i < 3; i++) {
        printf("Finishing queue %d.\n", i);
        err = clFinish(queues[i]);
        OCL_CHECK_ERR(err);
    }
    printf("Kernel execution complete\n");

    // Compute execution time
    for (int i = 0; i < 3; i++) {
        cl_ulong start, end;
        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &end, NULL);
        OCL_CHECK_ERR(err);
        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &start, NULL);
        OCL_CHECK_ERR(err);
        double time = 1.0e-9 * (end - start);
        printf("%.6f\n", time);
    }

    // We use the fourth queue to read data back.
    err = clEnqueueReadBuffer(queues[3], dev_c, CL_TRUE, 0,
                              n_bytes_c, c->data,
                              0, NULL, NULL);
    OCL_CHECK_ERR(err);
    tensor_check_equal(c, c_golden_blocked_reordered, LINALG_EPSILON);

    // Release OpenCL
    clReleaseMemObject(dev_a);
    clReleaseMemObject(dev_b);
    clReleaseMemObject(dev_c);

    for(int i = 0; i < 4; i++) {
        OCL_CHECK_ERR(clFlush(queues[i]));
        OCL_CHECK_ERR(clFinish(queues[i]));
        clReleaseCommandQueue(queues[i]);
    }
    for (int i = 0; i < 3; i++) {
        clReleaseKernel(kernels[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(ctx);

    // Free tensors
    tensor_free(a);
    tensor_free(a_blocked);
    tensor_free(b);
    tensor_free(b_transpose);
    tensor_free(b_transpose_blocked);
    tensor_free(c);
    tensor_free(c_ref);
    tensor_free(c_golden);
    tensor_free(c_golden_blocked);
    tensor_free(c_golden_blocked_reordered);
}
