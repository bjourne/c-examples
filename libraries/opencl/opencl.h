// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef OPENCL_H
#define OPENCL_H

#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms);
void ocl_get_devices(cl_platform_id platform,
                     cl_uint *n_devices, cl_device_id **devices);

void ocl_print_device_details(cl_device_id dev, int ind);
void ocl_check_err(cl_int err);

bool
ocl_load_kernel(cl_context ctx, cl_device_id dev, const char *fname,
                cl_program *program, cl_kernel *kernel);

void ocl_run_nd_kernel(cl_command_queue queue, cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t *global,
                       const size_t *local,
                       int n_args, ...);

void *ocl_get_platform_info(cl_platform_id platform,
                            cl_platform_info info);

#endif
