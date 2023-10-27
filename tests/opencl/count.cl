// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Count the number of elements divisible by 7, with and without SIMD.

// On my machine int16 is slighty faster than int8.
#define VECTOR_WIDTH 16
#include "libraries/opencl/utils.cl"

__kernel void
count_divisible(const uint N,
                const __global uint * restrict arr,
                __global ulong * restrict cnt) {

    uint gs = get_local_size(0);
    uint id = get_local_id(0);
    ranges r = slice_work(N, gs, id, 1);
    uint lcnt = 0;
    for (uint i = r.c0; i < r.c1; i++) {
        if (arr[i] % 7 == 0) {
            lcnt++;
        }
    }
    lcnt = work_group_reduce_add(lcnt);
    if (id == 0) {
        cnt[0] = lcnt;
    }
}

__kernel void
count_divisible_simd(const int N,
                     const __global int * restrict arr,
                     __global ulong * restrict cnt) {
    int gs = get_local_size(0);
    int id = get_local_id(0);

    ranges r = slice_work(N, gs, id, VECTOR_WIDTH);
    vint scnt = 0;
    int rem = 0;
    for (int i = r.c0; i < r.c1; i += VECTOR_WIDTH) {
        vint a = VLOAD_AT(i, arr);
        vint mask = (a % 7) == 0;
        scnt -= mask;
    }
    for (int j = r.c1; j < r.c2; j++) {
        if (arr[j] % 7 == 0) {
            rem++;
        }
    }
    int lcnt = 0;
    for (int i = 0; i < VECTOR_WIDTH; i++) {
        lcnt += scnt[i];
    }
    lcnt = work_group_reduce_add(lcnt);
    if (id == 0) {
        cnt[0] = lcnt;
    }
}
