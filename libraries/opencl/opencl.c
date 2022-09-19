// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"
#include "paths/paths.h"
#include "opencl.h"

#define CL_ERR_RETURN_STRING(x) case x: return #x;

const char *
err_str(cl_int err) {
    switch (err) {
        CL_ERR_RETURN_STRING(CL_SUCCESS                        )
        CL_ERR_RETURN_STRING(CL_DEVICE_NOT_FOUND               )
        CL_ERR_RETURN_STRING(CL_DEVICE_NOT_AVAILABLE           )
        CL_ERR_RETURN_STRING(CL_COMPILER_NOT_AVAILABLE         )
        CL_ERR_RETURN_STRING(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        CL_ERR_RETURN_STRING(CL_OUT_OF_RESOURCES               )
        CL_ERR_RETURN_STRING(CL_OUT_OF_HOST_MEMORY             )
        CL_ERR_RETURN_STRING(CL_PROFILING_INFO_NOT_AVAILABLE   )
        CL_ERR_RETURN_STRING(CL_MEM_COPY_OVERLAP               )
        CL_ERR_RETURN_STRING(CL_IMAGE_FORMAT_MISMATCH          )
        CL_ERR_RETURN_STRING(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        CL_ERR_RETURN_STRING(CL_BUILD_PROGRAM_FAILURE          )
        CL_ERR_RETURN_STRING(CL_MAP_FAILURE                    )
        CL_ERR_RETURN_STRING(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        CL_ERR_RETURN_STRING(CL_COMPILE_PROGRAM_FAILURE        )
        CL_ERR_RETURN_STRING(CL_LINKER_NOT_AVAILABLE           )
        CL_ERR_RETURN_STRING(CL_LINK_PROGRAM_FAILURE           )
        CL_ERR_RETURN_STRING(CL_DEVICE_PARTITION_FAILED        )
        CL_ERR_RETURN_STRING(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        CL_ERR_RETURN_STRING(CL_INVALID_VALUE                  )
        CL_ERR_RETURN_STRING(CL_INVALID_DEVICE_TYPE            )
        CL_ERR_RETURN_STRING(CL_INVALID_PLATFORM               )
        CL_ERR_RETURN_STRING(CL_INVALID_DEVICE                 )
        CL_ERR_RETURN_STRING(CL_INVALID_CONTEXT                )
        CL_ERR_RETURN_STRING(CL_INVALID_QUEUE_PROPERTIES       )
        CL_ERR_RETURN_STRING(CL_INVALID_COMMAND_QUEUE          )
        CL_ERR_RETURN_STRING(CL_INVALID_HOST_PTR               )
        CL_ERR_RETURN_STRING(CL_INVALID_MEM_OBJECT             )
        CL_ERR_RETURN_STRING(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CL_ERR_RETURN_STRING(CL_INVALID_IMAGE_SIZE             )
        CL_ERR_RETURN_STRING(CL_INVALID_SAMPLER                )
        CL_ERR_RETURN_STRING(CL_INVALID_BINARY                 )
        CL_ERR_RETURN_STRING(CL_INVALID_BUILD_OPTIONS          )
        CL_ERR_RETURN_STRING(CL_INVALID_PROGRAM                )
        CL_ERR_RETURN_STRING(CL_INVALID_PROGRAM_EXECUTABLE     )
        CL_ERR_RETURN_STRING(CL_INVALID_KERNEL_NAME            )
        CL_ERR_RETURN_STRING(CL_INVALID_KERNEL_DEFINITION      )
        CL_ERR_RETURN_STRING(CL_INVALID_KERNEL                 )
        CL_ERR_RETURN_STRING(CL_INVALID_ARG_INDEX              )
        CL_ERR_RETURN_STRING(CL_INVALID_ARG_VALUE              )
        CL_ERR_RETURN_STRING(CL_INVALID_ARG_SIZE               )
        CL_ERR_RETURN_STRING(CL_INVALID_KERNEL_ARGS            )
        CL_ERR_RETURN_STRING(CL_INVALID_WORK_DIMENSION         )
        CL_ERR_RETURN_STRING(CL_INVALID_WORK_GROUP_SIZE        )
        CL_ERR_RETURN_STRING(CL_INVALID_WORK_ITEM_SIZE         )
        CL_ERR_RETURN_STRING(CL_INVALID_GLOBAL_OFFSET          )
        CL_ERR_RETURN_STRING(CL_INVALID_EVENT_WAIT_LIST        )
        CL_ERR_RETURN_STRING(CL_INVALID_EVENT                  )
        CL_ERR_RETURN_STRING(CL_INVALID_OPERATION              )
        CL_ERR_RETURN_STRING(CL_INVALID_GL_OBJECT              )
        CL_ERR_RETURN_STRING(CL_INVALID_BUFFER_SIZE            )
        CL_ERR_RETURN_STRING(CL_INVALID_MIP_LEVEL              )
        CL_ERR_RETURN_STRING(CL_INVALID_GLOBAL_WORK_SIZE       )
        CL_ERR_RETURN_STRING(CL_INVALID_PROPERTY               )
        CL_ERR_RETURN_STRING(CL_INVALID_IMAGE_DESCRIPTOR       )
        CL_ERR_RETURN_STRING(CL_INVALID_COMPILER_OPTIONS       )
        CL_ERR_RETURN_STRING(CL_INVALID_LINKER_OPTIONS         )
        CL_ERR_RETURN_STRING(CL_INVALID_DEVICE_PARTITION_COUNT )
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

static
void
ocl_check_err_for_call(const char *func, cl_int err) {
    if (err == CL_SUCCESS)  {
        return;
    }
    printf("OpenCL call failed: %s\n", func);
    ocl_check_err(err);
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

void
ocl_get_platforms(cl_uint *n_platforms, cl_platform_id **platforms) {
    cl_int err;

    err = clGetPlatformIDs(0, NULL, n_platforms);
    ocl_check_err_for_call("clGetPlatformIDs", err);

    *platforms = (cl_platform_id *)malloc(
        sizeof(cl_platform_id) * *n_platforms);

    err = clGetPlatformIDs(*n_platforms, *platforms, NULL);
    ocl_check_err(err);
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
    for (int i = 0; i < ARRAY_SIZE(attr_types); i++) {
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
}

bool
ocl_load_kernel(cl_context ctx, cl_device_id dev, const char *fname,
                cl_program *program, cl_kernel *kernel) {

    FILE *fp = fopen(fname, "r");
    if (!fp) {
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *source = (char*)malloc(sizeof(char)*(size + 1));
    size_t n_bytes = size * sizeof(char);
    assert(fread(source, 1, n_bytes, fp) == n_bytes);
    source[size] = '\0';
    fclose(fp);

    size_t n_source = strlen(source);
    cl_int err;
    *program = clCreateProgramWithSource(
        ctx, 1,
        (const char **)&source,
        (const size_t *)&n_source, &err);
    ocl_check_err(err);

    err = clBuildProgram(*program, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build failed: %s\n", err_str(err));
        size_t n_bytes;
        err = clGetProgramBuildInfo(*program, dev, CL_PROGRAM_BUILD_LOG,
                                    0, NULL, &n_bytes);
        ocl_check_err(err);
        assert(n_bytes > 0);

        // Allocate memory for the log
        char *log = (char *) malloc(n_bytes);

        // Get the log
        clGetProgramBuildInfo(*program, dev, CL_PROGRAM_BUILD_LOG, n_bytes, log, NULL);

        // Print the log
        printf("%s\n", log);
        assert(false);
        return false;
    }
    free(source);

    char *stem = paths_stem(fname);
    *kernel = clCreateKernel(*program, stem, &err);
    free(stem);
    ocl_check_err(err);
    return true;
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
    // Divide by 2 here is lame.
    for (int i = 0; i < n_args / 2; i++) {
        size_t arg_size = va_arg(ap, size_t);
        void *arg_value = va_arg(ap, void *);
        err = clSetKernelArg(kernel, i, arg_size, arg_value);
        ocl_check_err(err);
    }
    va_end(ap);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
                                 global, local, 0, NULL, &event);
    ocl_check_err(err);
    err = clWaitForEvents(1, &event);
    ocl_check_err(err);
}
