// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// All functions should return a cl_int, indicating whether an error
// occurred.
#ifndef OPENCL_H
#define OPENCL_H

#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Some more error codes. All OpenCL's error codes are negative we can
// freely use positive values.
#define OCL_FILE_NOT_FOUND      1
#define OCL_BAD_PLATFORM_IDX    2
#define OCL_BAD_DEVICE_IDX      3

// Print functions
void
ocl_print_platform_details(cl_platform_id plat);

void
ocl_print_device_details(cl_device_id dev, int ind);

// Convenience functions
cl_int
ocl_basic_setup(
    cl_uint plat_idx, cl_uint dev_idx,
    cl_platform_id *platform,
    cl_device_id *device,
    cl_context *ctx,
    size_t n_queues,
    cl_command_queue *queues
);

// Platform and device functions
cl_int
ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms);
cl_int
ocl_get_devices(cl_platform_id platform,
                cl_uint *n_devices, cl_device_id **devices);
void *
ocl_get_platform_info(cl_platform_id platform,
                      cl_platform_info info);

// Error handling
void ocl_check_err(cl_int err);
void ocl_check_err2(cl_int err, char *file, int line);
#define OCL_CHECK_ERR(err) ocl_check_err2(err, __FILE__, __LINE__)

// Loading and running kernels
cl_int
ocl_load_kernels(cl_context ctx, cl_device_id dev, const char *path,
                 size_t n_kernels, char *names[],
                 cl_program *program, cl_kernel *kernels);

cl_int
ocl_set_kernel_arguments(cl_kernel kernel, int n_args, ...);

cl_int
ocl_run_nd_kernel(cl_command_queue queue, cl_kernel kernel,
                  cl_uint work_dim,
                  const size_t *global,
                  const size_t *local,
                  int n_args, ...);


// Buffer creation functions
cl_int
ocl_create_and_write_buffer(cl_context ctx, cl_mem_flags flags,
                            cl_command_queue queue,
                            void *src, size_t n_bytes,
                            cl_mem *mem);
cl_int
ocl_create_and_fill_buffer(cl_context ctx, cl_mem_flags flags,
                           cl_command_queue queue,
                           void *pattern, size_t pattern_size,
                           size_t n_bytes, cl_mem *mem);
cl_int
ocl_create_empty_buffer(cl_context ctx, cl_mem_flags flags,
                        size_t n_bytes, cl_mem *mem);

// Since we wrap writes, we also wrap reads.
cl_int
ocl_read_buffer(cl_command_queue queue, void *dst,
                size_t n_bytes, cl_mem mem);

// Pipe creation
cl_int
ocl_create_pipe(cl_context ctx,
                cl_uint packet_size, cl_uint n_packets,
                cl_mem *mem);

// Event handling
cl_int
ocl_poll_event_until(cl_event event, cl_int exec_status, cl_uint millis);

#endif
