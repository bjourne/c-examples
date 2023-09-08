// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"
#include "files/files.h"
#include "paths/paths.h"
#include "opencl.h"

#define ERR_RETURN_STRING(x) case x: return #x;

#define BOOL_TO_YES_NO(x) ((x) ? "yes" : "no")

const char *
err_str(cl_int err) {
    switch (err) {
        ERR_RETURN_STRING(CL_SUCCESS                        )
        ERR_RETURN_STRING(CL_DEVICE_NOT_FOUND               )
        ERR_RETURN_STRING(CL_DEVICE_NOT_AVAILABLE           )
        ERR_RETURN_STRING(CL_COMPILER_NOT_AVAILABLE         )
        ERR_RETURN_STRING(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        ERR_RETURN_STRING(CL_OUT_OF_RESOURCES               )
        ERR_RETURN_STRING(CL_OUT_OF_HOST_MEMORY             )
        ERR_RETURN_STRING(CL_PROFILING_INFO_NOT_AVAILABLE   )
        ERR_RETURN_STRING(CL_MEM_COPY_OVERLAP               )
        ERR_RETURN_STRING(CL_IMAGE_FORMAT_MISMATCH          )
        ERR_RETURN_STRING(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        ERR_RETURN_STRING(CL_BUILD_PROGRAM_FAILURE          )
        ERR_RETURN_STRING(CL_MAP_FAILURE                    )
        ERR_RETURN_STRING(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        ERR_RETURN_STRING(CL_COMPILE_PROGRAM_FAILURE        )
        ERR_RETURN_STRING(CL_LINKER_NOT_AVAILABLE           )
        ERR_RETURN_STRING(CL_LINK_PROGRAM_FAILURE           )
        ERR_RETURN_STRING(CL_DEVICE_PARTITION_FAILED        )
        ERR_RETURN_STRING(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        ERR_RETURN_STRING(CL_INVALID_VALUE                  )
        ERR_RETURN_STRING(CL_INVALID_DEVICE_TYPE            )
        ERR_RETURN_STRING(CL_INVALID_PLATFORM               )
        ERR_RETURN_STRING(CL_INVALID_DEVICE                 )
        ERR_RETURN_STRING(CL_INVALID_CONTEXT                )
        ERR_RETURN_STRING(CL_INVALID_QUEUE_PROPERTIES       )
        ERR_RETURN_STRING(CL_INVALID_COMMAND_QUEUE          )
        ERR_RETURN_STRING(CL_INVALID_HOST_PTR               )
        ERR_RETURN_STRING(CL_INVALID_MEM_OBJECT             )
        ERR_RETURN_STRING(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        ERR_RETURN_STRING(CL_INVALID_IMAGE_SIZE             )
        ERR_RETURN_STRING(CL_INVALID_SAMPLER                )
        ERR_RETURN_STRING(CL_INVALID_BINARY                 )
        ERR_RETURN_STRING(CL_INVALID_BUILD_OPTIONS          )
        ERR_RETURN_STRING(CL_INVALID_PROGRAM                )
        ERR_RETURN_STRING(CL_INVALID_PROGRAM_EXECUTABLE     )
        ERR_RETURN_STRING(CL_INVALID_KERNEL_NAME            )
        ERR_RETURN_STRING(CL_INVALID_KERNEL_DEFINITION      )
        ERR_RETURN_STRING(CL_INVALID_KERNEL                 )
        ERR_RETURN_STRING(CL_INVALID_ARG_INDEX              )
        ERR_RETURN_STRING(CL_INVALID_ARG_VALUE              )
        ERR_RETURN_STRING(CL_INVALID_ARG_SIZE               )
        ERR_RETURN_STRING(CL_INVALID_KERNEL_ARGS            )
        ERR_RETURN_STRING(CL_INVALID_WORK_DIMENSION         )
        ERR_RETURN_STRING(CL_INVALID_WORK_GROUP_SIZE        )
        ERR_RETURN_STRING(CL_INVALID_WORK_ITEM_SIZE         )
        ERR_RETURN_STRING(CL_INVALID_GLOBAL_OFFSET          )
        ERR_RETURN_STRING(CL_INVALID_EVENT_WAIT_LIST        )
        ERR_RETURN_STRING(CL_INVALID_EVENT                  )
        ERR_RETURN_STRING(CL_INVALID_OPERATION              )
        ERR_RETURN_STRING(CL_INVALID_GL_OBJECT              )
        ERR_RETURN_STRING(CL_INVALID_BUFFER_SIZE            )
        ERR_RETURN_STRING(CL_INVALID_MIP_LEVEL              )
        ERR_RETURN_STRING(CL_INVALID_GLOBAL_WORK_SIZE       )
        ERR_RETURN_STRING(CL_INVALID_PROPERTY               )
        ERR_RETURN_STRING(CL_INVALID_IMAGE_DESCRIPTOR       )
        ERR_RETURN_STRING(CL_INVALID_COMPILER_OPTIONS       )
        ERR_RETURN_STRING(CL_INVALID_LINKER_OPTIONS         )
        ERR_RETURN_STRING(CL_INVALID_DEVICE_PARTITION_COUNT )
    default:
        return NULL;
    }
}

void
ocl_check_err(cl_int err) {
    if (err == CL_SUCCESS)  {
        return;
    }
    const char *s = err_str(err);
    if (s) {
        printf("OpenCL error: %s\n", s);
    } else {
        printf("Unknown OpenCL error: %d\n", err);
    }
    assert(false);
}

static void
print_prefix(int ind) {
    for (int i = 0; i < ind; i++) {
        printf(" ");
    }
}

static void
print_device_info_str(cl_device_id dev, int attr, char *attr_name) {
    size_t n_bytes;
    clGetDeviceInfo(dev, attr, 0, NULL, &n_bytes);
    char *bytes = (char *)malloc(n_bytes);
    clGetDeviceInfo(dev, attr, n_bytes, bytes, NULL);
    printf("%-15s: %s\n", attr_name, bytes);
    free(bytes);
}

bool
ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms) {
    cl_int err;

    err = clGetPlatformIDs(0, NULL, n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        return false;
    }
    *platforms = (cl_platform_id *)malloc(
        sizeof(cl_platform_id) * *n_platforms);

    err = clGetPlatformIDs(*n_platforms, *platforms, NULL);
    ocl_check_err(err);
    return true;
}

void *
ocl_get_platform_info(cl_platform_id platform,
                      cl_platform_info info) {
    size_t n_bytes;
    cl_int err;
    err = clGetPlatformInfo(platform, info,
                            0, NULL, &n_bytes);
    ocl_check_err(err);
    void *ptr = (void *)malloc(n_bytes);
    clGetPlatformInfo(platform, info,
                      n_bytes, ptr, NULL);
    ocl_check_err(err);
    return ptr;
}

void
ocl_get_devices(cl_platform_id platform,
                cl_uint *n_devices, cl_device_id **devices) {

    cl_int err;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, n_devices);
    ocl_check_err(err);

    *devices = (cl_device_id *)malloc(
        sizeof(cl_device_id) * *n_devices);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                         *n_devices, *devices, NULL);
    ocl_check_err(err);
}

void
ocl_print_device_details(cl_device_id dev, int ind) {
    char *attr_names[] = {
        "Name", "Version", "Driver", "C Version"
    };
    int attr_types[] = {
        CL_DEVICE_NAME,
        CL_DEVICE_VERSION, CL_DRIVER_VERSION,
        CL_DEVICE_OPENCL_C_VERSION
    };
    for (size_t i = 0; i < ARRAY_SIZE(attr_types); i++) {
        print_prefix(ind);
        print_device_info_str(dev, attr_types[i], attr_names[i]);
    }

    print_prefix(ind);
    cl_uint n_compute_units;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(cl_uint), &n_compute_units, NULL);
    printf("%-15s: %d\n", "Compute units", n_compute_units);

    print_prefix(ind);
    cl_ulong n_mem;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &n_mem, NULL);
    printf("%-15s: %ld\n", "Global memory", n_mem);

    print_prefix(ind);
    cl_ulong n_alloc;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_ulong), &n_alloc, NULL);
    printf("%-15s: %ld\n", "Max allocation", n_alloc);

    print_prefix(ind);
    size_t n_bytes;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    0, NULL, &n_bytes);
    size_t *d = (size_t *)malloc(n_bytes);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, n_bytes, d, NULL);
    printf("%-15s: %ld, %ld, %ld\n", "Max work items", d[0], d[1], d[2]);
    free(d);

    cl_device_info flags[] = {
        CL_DEVICE_PIPE_SUPPORT,
        CL_DEVICE_COMPILER_AVAILABLE,
        CL_DEVICE_IMAGE_SUPPORT
    };
    char *names[] = {
        "Pipe support",
        "Compiler usable",
        "Image support"
    };

    for (size_t i = 0; i < ARRAY_SIZE(flags); i++) {
        print_prefix(ind);
        cl_bool val;
        cl_int err = clGetDeviceInfo(dev, flags[i],
                                     sizeof(cl_bool), &val, NULL);
        printf("%-15s: %s\n", names[i],
               BOOL_TO_YES_NO(err == CL_SUCCESS && val));
    }
}

bool
ocl_load_kernels(cl_context ctx, cl_device_id dev, const char *path,
                 int n_kernels, char *names[],
                 cl_program *program, cl_kernel *kernels) {
    char *data = NULL, *log = NULL;
    size_t n_data;
    cl_int err;
    if (!files_read(path, &data, &n_data)) {
        goto err;
    }

    // Check file extension
    if (!strcmp(paths_ext(path), "cl")) {
        *program = clCreateProgramWithSource(
            ctx, 1,
            (const char **)&data,
            (const size_t *)&n_data, &err);

    } else {
        *program = clCreateProgramWithBinary(
            ctx, 1, &dev,
            &n_data, (const unsigned char **)&data,
            &err, NULL);
    }
    ocl_check_err(err);

    err = clBuildProgram(*program, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build failed: %s\n", err_str(err));
        size_t n_log;
        err = clGetProgramBuildInfo(*program, dev, CL_PROGRAM_BUILD_LOG,
                                    0, NULL, &n_log);
        ocl_check_err(err);
        assert(n_log > 0);

        // Acquire, print, and free the log.
        log = (char *)malloc(n_log);
        clGetProgramBuildInfo(*program, dev,
                              CL_PROGRAM_BUILD_LOG, n_log, log, NULL);
        printf("%s\n", log);
        goto err;
    }

    for (int i = 0; i < n_kernels; i++) {
        kernels[i] = clCreateKernel(*program, names[i], &err);
        ocl_check_err(err);
    }
    free(data);
    return true;
 err:
    if (data) {
        free(data);
    }
    if (log) {
        free(log);
    }
    return false;
}


static void
set_kernel_arguments(cl_kernel kernel, int n_args, va_list ap) {
    // Divide by 2 here is lame.
    for (int i = 0; i < n_args / 2; i++) {
        size_t arg_size = va_arg(ap, size_t);
        void *arg_value = va_arg(ap, void *);
        cl_int err = clSetKernelArg(kernel, i, arg_size, arg_value);
        ocl_check_err(err);
    }
}

void
ocl_set_kernel_arguments(cl_kernel kernel, int n_args, ...) {
    va_list ap;
    va_start(ap, n_args);
    set_kernel_arguments(kernel, n_args, ap);
    va_end(ap);
}

void
ocl_run_nd_kernel(cl_command_queue queue, cl_kernel kernel,
                  cl_uint work_dim,
                  const size_t *global,
                  const size_t *local,
                  int n_args, ...) {
    va_list ap;
    cl_int err;

    va_start(ap, n_args);
    set_kernel_arguments(kernel, n_args, ap);
    va_end(ap);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL,
                                 global, local, 0, NULL, &event);
    ocl_check_err(err);
    err = clWaitForEvents(1, &event);
    ocl_check_err(err);
}
