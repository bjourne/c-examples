// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
// CUDA includes
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include "cud/cud.h"

#define RETURN_STRING(x) case x: return #x

static const char *
cud_core_err_str(cudaError_t st) {
    switch (st) {
        RETURN_STRING(cudaErrorMemoryAllocation);
        RETURN_STRING(cudaErrorInvalidDevice);
        RETURN_STRING(cudaErrorInitializationError);
    default:
        return NULL;
    }
}

void
cud_assert(cudaError_t st, char *file, int line) {
    if (st == cudaSuccess) {
        return;
    }
    const char *s = cud_core_err_str(st);
    printf("%-12s: %s (%d)\n", "Core error", s, st);
    printf("%-12s: %s:%d\n", "Caused by", file, line);
    abort();
}

#define KEY_WIDTH   "24"

void
cud_print_system_details() {
    int cnt;
    CUD_ASSERT(cudaGetDeviceCount(&cnt));

    printf("%-" KEY_WIDTH "s: %d\n", "N. of devices", cnt);
    for (int i = 0; i < cnt; i++) {
        struct cudaDeviceProp props;
        CUD_ASSERT(cudaGetDeviceProperties(&props, i));
        printf("%-" KEY_WIDTH "s: %d.%d\n",
               "Compute cap.", props.major, props.minor);
        printf("%-" KEY_WIDTH "s: %d\n",
               "Max n. of thr./block",
               props.maxThreadsPerBlock);
        printf("%-" KEY_WIDTH "s: %d\n",
               "Max n. of thr./mproc.",
               props.maxThreadsPerMultiProcessor);
    }
}
