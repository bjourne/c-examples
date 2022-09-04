// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Benchmarks 8x8 dct on tensors
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/common.h"
#include "opencl/opencl.h"
#include "tensors/tensors.h"

#define SQRT2 (1.4142135623730951f)

// sin(pi/16) - cos(pi/16)
#define c1_a (-0.78569495838710218127789736765721679604624211314138128724)
// -(sin(pi/16) + cos(pi/16))
#define c1_b (-1.17587560241935871697446710461126127790162534864529090275)
// cos(pi/16)
#define c1_c (0.980785280403230449126182236134239036973933730893336095002)
// sin(3pi/16) - cos(3pi/16)
#define c3_a (-0.27589937928294301233595756366937288236362362123244591752)
// -(sin(3pi/16) + cos(3pi/16))
#define c3_b (-1.38703984532214746182161919156643863111349800274205400937)
// cos(3pi/16)
#define c3_c (0.831469612302545237078788377617905756738560811987249963446)
// sqrt(2) * (sin(6pi/16) - cos(6pi/16))
#define c6_a (0.765366864730179543456919968060797733522689124971254082867)
// sqrt(2) * -(sin(6pi/16) + cos(6pi/16))
#define c6_b (-1.84775906502257351225636637879357657364483325172728497223)
// sqrt(2) * cos(6pi/16)
#define c6_c (0.541196100146196984399723205366389420061072063378015444681)

// 1/sqrt(8)
#define C_NORM      (0.35355339059327373f)

// 1/2
#define C_NORM2     0.5f

//d = sqrt(2) * cos(5 * pi / 16)
#define C_A 1.3870398453221474618216191915664f
#define C_B 1.3065629648763765278566431734272f
#define C_C 1.1758756024193587169744671046113f
#define C_D 0.78569495838710218127789736765722f
#define C_E 0.54119610014619698439972320536639f
#define C_F 0.27589937928294301233595756366937f
#define DCT8_NORM_NVIDIA 0.35355339059327376220042218105242f

static void
loeffler8(float x[8], float y[8]) {

    // Stage 1 reflectors
    float s10 = x[0] + x[7];
    float s11 = x[1] + x[6];
    float s12 = x[2] + x[5];
    float s13 = x[3] + x[4];

    float s14 = x[3] - x[4];
    float s15 = x[2] - x[5];
    float s16 = x[1] - x[6];
    float s17 = x[0] - x[7];

    // Stage 2 stuff
    float s20 = s10 + s13;
    float s21 = s11 + s12;
    float s22 = s11 - s12;
    float s23 = s10 - s13;

    // Rotations?
    float c3_rot_tmp = c3_c * (s14 + s17);
    float c1_rot_tmp = c1_c * (s15 + s16);

    float s24 = c3_a * s17 + c3_rot_tmp;
    float s25 = c1_a * s16 + c1_rot_tmp;
    float s26 = c1_b * s15 + c1_rot_tmp;
    float s27 = c3_b * s14 + c3_rot_tmp;

    // Stage 3
    float c6_rot_tmp = c6_c * (s22 + s23);

    float s30 = s20 + s21;
    float s31 = s20 - s21;
    float s34 = s24 + s26;
    float s35 = s27 - s25;
    float s36 = s24 - s26;
    float s37 = s27 + s25;

    y[0] = C_NORM * s30;
    y[1] = C_NORM * (s37 + s34);
    y[2] = C_NORM * (c6_a * s23 + c6_rot_tmp);
    y[3] = C_NORM2 * s35;
    y[4] = C_NORM * s31;
    y[5] = C_NORM2 * s36;
    y[6] = C_NORM * (c6_b * s22 + c6_rot_tmp);
    y[7] = C_NORM * (s37 - s34);
}

// Nvidia's variant of the algorithm.
static void
loeffler8_nvidia(float x[8], float y[8]) {

    // Stage 1 reflectors
    float s10 = x[0] + x[7];
    float s11 = x[1] + x[6];
    float s12 = x[2] + x[5];
    float s13 = x[3] + x[4];

    float s14 = x[0] - x[7];
    float s15 = x[2] - x[5];
    float s16 = x[4] - x[3];
    float s17 = x[6] - x[1];

    // Stage 2 things
    float s20 = s10 + s13;
    float s21 = s10 - s13;
    float s22 = s11 + s12;
    float s23 = s11 - s12;

    float norm = DCT8_NORM_NVIDIA;

    y[0] = norm * (s20 + s22);
    y[2] = norm * (C_B * s21 + C_E * s23);
    y[4] = norm * (s20 - s22);
    y[6] = norm * (C_E * s21 - C_B * s23);

    y[1] = norm * (C_A * s14 - C_C * s17 + C_D * s15 - C_F * s16);
    y[3] = norm * (C_C * s14 + C_F * s17 - C_A * s15 + C_D * s16);
    y[5] = norm * (C_D * s14 + C_A * s17 + C_F * s15 - C_C * s16);
    y[7] = norm * (C_F * s14 + C_D * s17 + C_C * s15 + C_A * s16);
}

int
main(int argc, char *argv[]) {

    const int IMAGE_WIDTH = 256;
    const int IMAGE_HEIGHT = 256;
    const int IMAGE_N_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float);

    const int BLOCKDIM_X = 8;
    const int BLOCKDIM_Y = 8;
    const int BLOCK_SIZE = 8;

    //  Single floats
    const int SIMD_LOC = 1;

    float x[8] = {20, 9, 10, 11, 12, 13, 14, 15};
    float y[8], y2[8];
    loeffler8_nvidia(x, y);
    loeffler8(x, y2);
    for (int i = 0; i < 8; i++) {
        printf("%2d %5.2f %5.2f\n", i, y[i], y2[i]);
    }
    printf("\n");

    // Setup OpenCL
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
    ocl_check_err(err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, dev, 0, &err);
    ocl_check_err(err);

    // Load kernel
    cl_program program;
    cl_kernel kernel;
    printf("* Loading kernel\n");
    ocl_load_kernel(ctx, dev, "libraries/opencl/dct8x8.cl",
                    &program, &kernel);

    // Allocate and initialize tensors
    printf("* Initializing tensors\n");
    int dims[] = {IMAGE_HEIGHT, IMAGE_WIDTH};
    tensor *image = tensor_init(2, dims);
    tensor *ref = tensor_init(2, dims);
    tensor *output = tensor_init(2, dims);
    tensor_randrange(image, 100);

    // Compute reference results
    tensor_dct2d_blocked(image, ref, 8, 8);

    // Allocate and write to OpenCL buffers
    cl_mem mem_image = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      IMAGE_N_BYTES, NULL, &err);
    ocl_check_err(err);
    cl_mem mem_dct = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                    IMAGE_N_BYTES, NULL, &err);
    ocl_check_err(err);
    err = clEnqueueWriteBuffer(queue, mem_image, CL_TRUE,
                               0, IMAGE_N_BYTES, image->data,
                               0, NULL, NULL);
    ocl_check_err(err);

    // Run kernel
    printf("* Running kernel\n");
    size_t local[] = {
        BLOCKDIM_Y / BLOCK_SIZE,
        BLOCKDIM_X / SIMD_LOC
    };
    size_t global[] = {
        IMAGE_HEIGHT / BLOCKDIM_Y * local[0],
        IMAGE_WIDTH / BLOCKDIM_X * local[1]
    };

    uint64_t start = nano_count();
    for (int i = 0; i < 1000; i++) {
        ocl_run_nd_kernel(queue, kernel,
                          2, global, local,
                          8,
                          sizeof(cl_mem), (void *)&mem_image,
                          sizeof(cl_mem), (void *)&mem_dct,
                          sizeof(cl_uint), (void *)&IMAGE_HEIGHT,
                          sizeof(cl_uint), (void *)&IMAGE_WIDTH);
    }

    uint64_t end = nano_count();
    double secs = (double)(end - start) / 1000 / 1000 / 1000;
    printf("\\--> %.3f seconds\n", secs);

    // Read data
    printf("* Reading device data\n");
    err = clEnqueueReadBuffer(queue, mem_dct, CL_TRUE,
                              0, IMAGE_N_BYTES, output->data,
                              0, NULL, NULL);
    ocl_check_err(err);

    /* printf("* Input:\n"); */
    /* tensor_print(image, "%4.0f", false); */
    /* printf("* Reference:\n"); */
    /* tensor_print(ref, "%4.0f", false); */
    /* printf("* Output:\n"); */
    /* tensor_print(output, "%4.0f", false); */

    tensor_check_equal(output, ref, 0.01);

    // Free tensors
    tensor_free(image);
    tensor_free(ref);
    tensor_free(output);

    // Teardown OpenCL

    // Release queue
    clFlush(queue);
    clFinish(queue);
    clReleaseCommandQueue(queue);

    // Free OpenCL memory
    clReleaseMemObject(mem_image);
    clReleaseMemObject(mem_dct);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(ctx);
    free(platforms);
    free(devices);

    return 0;
}
