// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Lists information about found OpenCL platforms. Ideas and concepts
// from https://gist.github.com/courtneyfaulkner/7919509
#include <stdio.h>
#include <stdlib.h>
#include "opencl/opencl.h"

static void
list_platforms() {
    cl_uint n_platforms;
    cl_platform_id *platforms;
    if (!ocl_get_platforms(&n_platforms, &platforms)) {
        printf("No OpenCL platforms found!\n");
        return;
    }

    for (cl_uint i  = 0; i < n_platforms; i++) {
        ocl_print_platform_details(platforms[i]);
    }
    free(platforms);
}

int
main(int argc, char *argv[]) {
    list_platforms();
}
