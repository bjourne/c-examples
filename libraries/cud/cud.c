// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
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
        RETURN_STRING(cudaErrorNotPermitted);
        RETURN_STRING(cudaErrorNotSupported);
        RETURN_STRING(cudaErrorSystemNotReady);
        RETURN_STRING(cudaErrorSystemDriverMismatch);
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
    pp->n_decimals = 2;
    pp->key_width = 18;

    int rt;
    CUD_ASSERT(cudaRuntimeGetVersion(&rt));
    int rt_maj = rt / 1000;
    int rt_min = (rt - rt_maj * 1000) / 10;

    size_t free, total;
    CUD_ASSERT(cudaMemGetInfo(&free, &total));

    pp_print_key_value(pp, "Runtime", "%d.%d", rt_maj, rt_min);
    pp_print_key_value_with_unit(
        pp, "Total memory",
        total, "B");
    pp_print_key_value_with_unit(
        pp, "Free memory",
        free, "B");
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
            pp, "Max thr./block",
            "%d", props.maxThreadsPerBlock);
        pp_print_key_value(
            pp, "Max thr./mproc.",
            "%d", props.maxThreadsPerMultiProcessor);
        pp_print_key_value_with_unit(pp, "Mem. clock", cr * 1000.0, "Hz");
        pp->n_decimals = 0;
        pp_print_key_value_with_unit(pp, "Bus width", bw, "bit");
        pp->n_decimals = 2;
        pp_print_key_value_with_unit(
            pp, "Max mem. bw",
            1000.0 * 2.0 * cr * (bw / 8), "B/s");

        pp_print_key_value_with_unit(
            pp, "Global memory",
            gm, "B");

        pp_print_key_value_with_unit(
            pp, "Shared mem./block",
            props.sharedMemPerBlock, "B");
    }
    pp->indent--;
    pp_free(pp);
}
