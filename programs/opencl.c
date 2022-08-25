// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Lists information about found OpenCL platforms. Ideas and concepts
// from https://gist.github.com/courtneyfaulkner/7919509
#include <stdio.h>
#include <stdlib.h>
#include "opencl/opencl.h"

static void
list_platforms() {
    const char *attr_names[] = {
        "Name", "Vendor",
        "Version", "Profile", "Extensions"
    };
    const cl_platform_info attr_types[] = {
        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
        CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS
    };

    cl_uint n_platforms;
    cl_platform_id *platforms;
    ocl_get_platforms(&n_platforms, &platforms);

    for (int i  = 0; i < n_platforms; i++) {
        for (int j = 0; j < 5; j++) {
            size_t n_bytes;
            clGetPlatformInfo(platforms[i], attr_types[j], 0, NULL, &n_bytes);
            char *info = (char *)malloc(n_bytes);
            clGetPlatformInfo(platforms[i], attr_types[j], n_bytes, info, NULL);
            printf("%-15s: %s\n", attr_names[j], info);
            free(info);
        }
        cl_uint n_devices;
        cl_device_id *devices;

        ocl_get_devices(platforms[i], &n_devices, &devices);
        printf("%-15s: %d\n", "Devices", n_devices);
        for (int j = 0; j < n_devices; j++) {
            ocl_print_device_details(devices[j], 2);
        }
        free(devices);
        printf("\n");
    }

    free(platforms);
}

int
main(int argc, char *argv[]) {
    list_platforms();
}
