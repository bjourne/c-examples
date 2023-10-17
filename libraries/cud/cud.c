// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
// CUDA includes
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include "cud/cud.h"
#include "pretty/pretty.h"

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

// See https://stackoverflow.com/questions/14800009/how-to-get-properties-from-active-cuda-device
void
cud_print_system_details() {
    int cnt;
    CUD_ASSERT(cudaGetDeviceCount(&cnt));

    pretty_printer *pp = pp_init();
    pp->key_width = 18;

    pp_print_key_value(pp, "N. of devices", "%d", cnt);
    printf("\n");
    pp->indent++;
    for (int i = 0; i < cnt; i++) {
        struct cudaDeviceProp props;
        CUD_ASSERT(cudaGetDeviceProperties(&props, i));
        int cr = props.memoryClockRate;
        int bw = props.memoryBusWidth;
        size_t gm = props.totalGlobalMem;
        pp_print_key_value(
            pp, "Name",
            "%s", props.name);
        pp_print_key_value(
            pp, "Compute cap.",
            "%d.%d", props.major, props.minor);
        pp_print_key_value(
            pp, "Warp size",
            "%d", props.warpSize);
        pp_print_key_value(
            pp, "Mem. clock",
            "%d MHz", cr / 1000);
        pp_print_key_value(
            pp, "Bus width",
            "%d bit", bw);
        pp_print_key_value(
            pp, "Max mem. bw",
            "%.1f GB/s", 2.0 * cr * (bw / 8) / 1.0e6);
        pp_print_key_value(
            pp, "Global memory",
            "%.2f GB", (double)gm / (1000.0 * 1000.0 * 1000.0));
        pp_print_key_value(
            pp, "Shared mem./block",
            "%.1f kB", (double)props.sharedMemPerBlock / 1000.0);
        pp_print_key_value(
            pp, "Max thr./block",
            "%d", props.maxThreadsPerBlock);
        pp_print_key_value(
            pp, "Max thr./mproc.",
            "%d", props.maxThreadsPerMultiProcessor);
    }
    pp->indent--;
    pp_free(pp);
}
