// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "opencl/opencl.h"
#include "tensors/tensors.h"

#define PE_ROWS                  2
#define PE_COLS                  2

#define DOT_PROD_VECTOR_SIZE     8
#define SCALING_FACTOR 32

#define ROWS_INTERLEAVED         32
#define COLUMNS_INTERLEAVED      32

#define MAT_A_BLOCK_WIDTH           (16 * DOT_PROD_VECTOR_SIZE)
#define MAT_A_BLOCK_HEIGHT          (ROWS_INTERLEAVED   * PE_ROWS)

#define MAT_B_BLOCK_HEIGHT          MAT_A_BLOCK_WIDTH
#define MAT_B_BLOCK_WIDTH           (COLUMNS_INTERLEAVED * PE_COLS)

#define HA (4 * MAT_A_BLOCK_HEIGHT)             // Matrix A height
#define WA (SCALING_FACTOR * MAT_A_BLOCK_WIDTH) // Matrix A width

#define HB WA                                   // Matrix B height
#define WB (4 * MAT_B_BLOCK_WIDTH)              // Matrix B width

#define HC HA                                   // Matrix C height
#define WC WB                                   // Matrix C width

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

int
main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s emu|fpga\n", argv[0]);
        exit(1);
    }
    bool emu = !strncmp(argv[1], "emu", strlen("emu"));

    tensor *a = tensor_init(2, (int[]){HA, WA});
    tensor *a_blocked = tensor_init(2, (int[]){HA, WA});
    tensor *b_transpose = tensor_init(2, (int[]){WB, HB});
    tensor *b_transpose_blocked = tensor_init(2, (int[]){WB, HB});
    tensor *b = tensor_init(2, (int[]){HB, WB});
    tensor *c = tensor_init(2, (int[]){HC, WC});
    tensor *c_ref = tensor_init(2, (int[]){HC, WC});

    printf("a = [%d, %d], b = [%d, %d], "
           "c = [%d, %d], a_blocked = [%d, %d], b_transpose = [%d, %d]\n",
           HA, WA, HB, WB, HC, WC, HA, WA, WB, HB);
    tensor_randrange(a, 10);
    tensor_randrange(b, 10);

    tensor_multiply(a, b, c_ref);
    block_wise_reformat(a->data, a_blocked->data, HA, WA,
                        MAT_A_BLOCK_HEIGHT, MAT_A_BLOCK_WIDTH);
    tensor_transpose(b, b_transpose);
    block_wise_reformat(b_transpose->data, b_transpose_blocked->data, WB, HB,
                        MAT_B_BLOCK_WIDTH, MAT_B_BLOCK_HEIGHT);

    cl_platform_id platform_id = platform_by_needle("Intel");
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
    const char *aocx_file = emu
        ? "matrix_mult_emu.aocx"
        : "matrix_mult_fpga.aocx";

    FILE *fp = fopen(aocx_file, "rb");
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
}
