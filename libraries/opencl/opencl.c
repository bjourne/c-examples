// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "datatypes/common.h"
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
        default: return "Unknown OpenCL error code";
    }
}

static void
check_err(cl_uint err) {
    if (err == CL_SUCCESS)  {
        return;
    }
    printf("OpenCL error: %s\n", err_str(err));
    assert(false);
}

static void
print_prefix(int ind) {
    for (int i = 0; i < ind; i++) {
        printf(" ");
    }
}

static void
print_device_info(cl_device_id dev, int attr, char *attr_name) {
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
    check_err(err);

    *platforms = (cl_platform_id *)malloc(
        sizeof(cl_platform_id) * *n_platforms);

    err = clGetPlatformIDs(*n_platforms, *platforms, NULL);
    check_err(err);
}

void
ocl_print_device_details(cl_device_id dev, int ind) {
    char *attr_names[] = {"Name", "Version", "Driver", "C Version"};
    int attr_types[] = {
        CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DRIVER_VERSION,
        CL_DEVICE_OPENCL_C_VERSION
    };
    for (int i = 0; i < ARRAY_SIZE(attr_types); i++) {
        print_prefix(ind);
        print_device_info(dev, attr_types[i], attr_names[i]);
    }
    cl_uint n_compute_units;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(cl_uint), &n_compute_units, NULL);
    print_prefix(ind);
    printf("%-15s: %d\n", "Compute units", n_compute_units);
}
