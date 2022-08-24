// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Lists information about found OpenCL platforms. Ideas and concepts
// from https://gist.github.com/courtneyfaulkner/7919509
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define LIST_SIZE 1024

static int ind = 0;

static void
print_prefix() {
    for (int i = 0; i < ind; i++) {
        printf(" ");
    }
}

static void
print_device_info(cl_device_id id, int attr, const char *attr_name) {
    size_t n_bytes;
    clGetDeviceInfo(id, attr, 0, NULL, &n_bytes);
    char *bytes = (char *)malloc(n_bytes);
    clGetDeviceInfo(id, attr, n_bytes, bytes, NULL);
    print_prefix();
    printf("%-15s: %s\n", attr_name, bytes);
    free(bytes);
}


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
    clGetPlatformIDs(0, NULL, &n_platforms);

    cl_platform_id *platforms = (cl_platform_id *)malloc(
        sizeof(cl_platform_id) * n_platforms);

    clGetPlatformIDs(n_platforms, platforms, NULL);
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
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);

        cl_device_id *devices = (cl_device_id *)malloc(
            sizeof(cl_device_id) * n_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, n_devices, devices, NULL);
        printf("%-15s: %d\n", "Devices", n_devices);
        ind += 2;
        for (int j = 0; j < n_devices; j++) {
            cl_device_id id = devices[j];
            print_device_info(id, CL_DEVICE_NAME, "Name");
            print_device_info(id, CL_DEVICE_VERSION, "Version");
            print_device_info(id, CL_DRIVER_VERSION, "Driver");
            print_device_info(id, CL_DEVICE_OPENCL_C_VERSION, "C Version");

            cl_uint n_compute_units;
            clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint), &n_compute_units, NULL);
            print_prefix();
            printf("%-15s: %d\n", "Compute units", n_compute_units);

        }
        ind -= 2;
        free(devices);
        printf("\n");
    }

    free(platforms);
}

int
main(int argc, char *argv[]) {
    list_platforms();
}
