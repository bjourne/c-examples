// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef OPENCL_H
#define OPENCL_H

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms);

void ocl_print_device_details(cl_device_id dev, int ind);

#endif
