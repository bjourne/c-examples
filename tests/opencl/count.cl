// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Count the number of elements divisible by 7, with and without SIMD.

// On my machine int16 is slighty faster than int8.
#define VECTOR_WIDTH 8
#include "libraries/opencl/utils.cl"

// It appears that interleaving accesses is more efficient on scalar
// kernels. At least it is on my GPU.
__kernel void
count_divisible(const uint N,
                const __global uint * restrict arr,
                __global uint * restrict cnt) {
    uint gs = get_local_size(0);
    uint id = get_local_id(0);
    uint lcnt = 0;
    for (uint i = id; i < N; i += gs) {
        lcnt += arr[i] % 7 == 0 ? 1 : 0;
    }
    cnt[0] = work_group_reduce_add(lcnt);
}

__kernel void
count_divisible_simd(const int N,
                     const __global int * restrict arr,
                     __global uint * restrict cnt) {
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
    int lcnt = rem;
    for (int i = 0; i < VECTOR_WIDTH; i++) {
        lcnt += scnt[i];
    }
    cnt[0] = work_group_reduce_add(lcnt);
}

// But it is not more efficient on SIMD kernels.
__kernel void
count_divisible_simd2(const int N,
                      const __global int * restrict arr,
                      __global uint * restrict cnt) {
    int gs = get_local_size(0);
    int id = get_local_id(0);
    uint n_chunks = N / VECTOR_WIDTH;

    // First do all chunks
    vint scnt = 0;
    for (uint i = id; i < n_chunks; i += gs) {
        vint a = VLOAD_AT(i * VECTOR_WIDTH, arr);
        vint mask = (a % 7) == 0;
        scnt -= mask;
    }
    uint lcnt = 0;
    for (uint i = n_chunks * VECTOR_WIDTH + id; i < N; i += gs) {
        if (arr[i] % 7 == 0) {
            lcnt++;
        }
    }
    for (int i = 0; i < VECTOR_WIDTH; i++) {
        lcnt += scnt[i];
    }
    cnt[0] = work_group_reduce_add(lcnt);
}
