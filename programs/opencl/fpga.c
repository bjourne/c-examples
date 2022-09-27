// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"
#include "opencl/opencl.h"
#include "tensors/tensors.h"

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

static void
block_wise_reformat(float * restrict src, float * restrict dst,
                    int height, int width,
                    int block_height, int block_width) {
    for(int i = 0; i < height; i += block_height) {
        for(int j = 0; j < width; j += block_width) {
            for(int k = 0; k < block_height; k++) {
                for(int l = 0; l < block_width; l++) {
                    *dst++ = src[(i + k) * width + (j + l)];
                }
            }
        }
    }
}

static cl_platform_id
platform_by_needle(const char *needle) {
    cl_uint n_platforms;
    cl_platform_id *platforms;
    ocl_get_platforms(&n_platforms, &platforms);

    cl_platform_id id = NULL;
    for (int i = 0; i < n_platforms; i++) {
        id = platforms[i];
        char *name = (char *)ocl_get_platform_info(id, CL_PLATFORM_NAME);
        if (strstr(name, needle)) {
            free(name);
            break;
        }
        free(name);
    }
    free(platforms);
    return id;
}

static cl_device_id
device_first(cl_platform_id platform_id) {
    cl_uint n_devices;
    cl_device_id *devices;

    ocl_get_devices(platform_id, &n_devices, &devices);
    cl_device_id device_id = devices[0];
    free(devices);
    return device_id;
}

static void
compute_gold_blocked(float* C, const float* A, const float* B,
                     unsigned int hA, unsigned int wA,
                     unsigned int wB, unsigned int hB)
{
    const int block_size = COMPUTE_GOLD_BLOCK_SIZE;
    for(unsigned int i0 = 0; i0 < hA ; i0 += block_size) {
        for(unsigned int j0 = 0; j0 < wB; j0 += block_size) {
            for(unsigned int k0=0; k0 < wA ; k0 += block_size ) {
                for(unsigned int i = i0; i < MIN(hA, i0 + block_size); i++) {
                    for(unsigned int j = j0; j < MIN(wB, j0 + block_size); j++) {
                        double sum = 0;
                        for(unsigned int k = k0; k < MIN(wA, k0 + block_size); k++) {
                            double a = A[i * wA + k];
                            double b = B[j * hB + k]; // B is transposed
                            sum += a * b;
                        }
                        C[i * wB + j] += (float)sum;
                    }
                }
            }
        }
    }
}

static  void
reorder_within_blocks(float * src,
                      float * dst,
                      int mat_height, int mat_width,
                      int num_sys_arr_columns, int block_width) {
    int num_elems = mat_height*mat_width;
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
}

int
main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s emu|fpga kernel-path\n", argv[0]);
        exit(1);
    }
    bool emu = !strncmp(argv[1], "emu", strlen("emu"));

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
    tensor_fill_rand_ints(a, 10);
    tensor_fill_rand_ints(b, 10);
    tensor_fill_const(c, 0);

    printf("** Multiplying on CPU**\n");
    tensor_multiply(a, b, c_ref);
    block_wise_reformat(a->data, a_blocked->data, HA, WA,
                        MAT_A_BLOCK_HEIGHT, MAT_A_BLOCK_WIDTH);
    tensor_transpose(b, b_transpose);
    block_wise_reformat(b_transpose->data, b_transpose_blocked->data, WB, HB,
                        MAT_B_BLOCK_WIDTH, MAT_B_BLOCK_HEIGHT);

    // Golden compute

    compute_gold_blocked(c_golden->data,
                         a->data, b_transpose->data,
                         HA_trim, WA, WB_trim, HB);
    block_wise_reformat(c_golden->data, c_golden_blocked->data,
                        HC, WC, MAT_C_BLOCK_HEIGHT, MAT_C_BLOCK_WIDTH);
    reorder_within_blocks(c_golden_blocked->data,
                          c_golden_blocked_reordered->data,
                          HC, WC, PE_COLS,
                          MAT_C_BLOCK_WIDTH);

    printf("** Setting up OpenCL **\n");
    cl_platform_id platform_id = platform_by_needle(
        emu ? "FPGA Emulation" : "FPGA SDK");
    assert(platform_id);

    cl_device_id dev_id = device_first(platform_id);
    assert(dev_id);
    ocl_print_device_details(dev_id, 0);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &err);
    ocl_check_err(err);

    // Create four queues
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    cl_command_queue queues[4];
    for (int i = 0; i < 4; i++) {
        queues[i] = clCreateCommandQueueWithProperties(ctx, dev_id,
                                                       props, &err);
        ocl_check_err(err);
    }

    printf("Creating device buffers\n");
    size_t n_bytes_a = HA * WA * sizeof(float);
    size_t n_bytes_b = HB * WB * sizeof(float);
    size_t n_bytes_c = HC * WC * sizeof(float);

    cl_mem dev_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n_bytes_a,
                                  NULL, &err);
    ocl_check_err(err);
    cl_mem dev_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n_bytes_b,
                                  NULL, &err);
    ocl_check_err(err);
    cl_mem dev_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n_bytes_c,
                                  NULL, &err);

    printf("Writing to device buffers\n");
    err = clEnqueueWriteBuffer(queues[0], dev_a, CL_TRUE, 0, n_bytes_a,
                               a_blocked->data, 0, NULL, NULL);
    ocl_check_err(err);

    err = clEnqueueWriteBuffer(queues[1], dev_b, CL_TRUE, 0, n_bytes_a,
                               b_transpose_blocked->data, 0, NULL, NULL);
    ocl_check_err(err);

    // Load the AOCX file.
    printf("Loading AOCX file\n");
    FILE *fp = fopen(argv[2], "rb");
    if (!fp) {
        perror("Error");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t length = ftell(fp);
    rewind(fp);

    char *binary = (char *)malloc(sizeof(char) * length);

    assert(fread((void *)binary, length, 1, fp) > 0);
    fclose(fp);

    // Create program from binary
    cl_program program = clCreateProgramWithBinary(ctx,
        1, &dev_id,
        &length, (const unsigned char **)&binary,
        &err, NULL);
    ocl_check_err(err);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    ocl_check_err(err);

    // Create the three kernels.
    const char *names[] = {"loadA", "loadB", "store"};
    cl_kernel kernels[3];
    for(int i = 0; i < 3; i++) {
        kernels[i] = clCreateKernel(program, (const char*)names[i], &err);
        ocl_check_err(err);
    }

    // LoadA kernel
    unsigned int mat_a_num_vectors_in_row_of_blocks =
        MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS;
    unsigned char mat_a_num_blocks_in_col = MAT_A_NUM_BLOCKS_IN_COL;
    unsigned char mat_b_num_blocks_in_row = MAT_B_NUM_BLOCKS_IN_ROW;

    ocl_set_kernel_arguments(
        kernels[0], 8,
        sizeof(cl_mem), (void *)&dev_a,
        sizeof(unsigned int), (void *)&mat_a_num_vectors_in_row_of_blocks,
        sizeof(unsigned char), (void *)&mat_a_num_blocks_in_col,
        sizeof(unsigned char), (void *)&mat_b_num_blocks_in_row);

    // LoadB kernel
    unsigned int mat_b_num_vectors_in_col_of_blocks =
        MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS;
    unsigned int mat_b_num_vectors_in_matrix =
        MAT_B_NUM_VECTORS_IN_MATRIX;

    ocl_set_kernel_arguments(
        kernels[1], 8,
        sizeof(cl_mem), (void *)&dev_b,
        sizeof(unsigned int), (void *)&mat_b_num_vectors_in_col_of_blocks,
        sizeof(unsigned int), (void *)&mat_b_num_vectors_in_matrix,
        sizeof(unsigned char), (void *)&mat_a_num_blocks_in_col);

    // Store kernel
    int mat_c_num_coalesced_words = WC * HC / PE_COLS;

    ocl_set_kernel_arguments(
        kernels[2], 4,
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
        ocl_check_err(err);
    }
    printf("Running kernels\n");
    for(int i=0; i < 3; i++) {
        err = clFlush(queues[i]);
        ocl_check_err(err);
    }

    for(int i = 0; i < 3; i++) {
         err = clFinish(queues[i]);
         ocl_check_err(err);
    }
    printf("Kernel execution complete\n");

    // Compute execution time
    for (int i = 0; i < 3; i++) {
        cl_ulong start, end;
        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &end, NULL);
        ocl_check_err(err);
        err = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &start, NULL);
        ocl_check_err(err);
        double time = 1.0e-9 * (end - start);
        printf("%.6f\n", time);
    }

    // We use the fourth queue to read data back.
    err = clEnqueueReadBuffer(queues[3], dev_c, CL_TRUE, 0,
                              n_bytes_c, c->data,
                              0, NULL, NULL);
    ocl_check_err(err);

    // Print some floats from c
    printf("%20s", "From device: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c->data[i]);
    }
    printf("\n");
    printf("%20s", "From cpu: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c_golden_blocked_reordered->data[i]);
    }
    printf("\n");

    // Release OpenCL
    clReleaseMemObject(dev_a);
    clReleaseMemObject(dev_b);
    clReleaseMemObject(dev_c);

    for(int i = 0; i < 4; i++) {
        ocl_check_err(clFlush(queues[i]));
        ocl_check_err(clFinish(queues[i]));
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
