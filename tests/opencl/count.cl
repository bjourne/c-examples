// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Count the number of elements divisible by 7, with and without SIMD.

// On my machine int16 is slighty faster than int8.
#define SW  16
#define VLOAD(ofs, arr) vload16((ofs), (arr))
#define ALIGN_TO(n, w)  (((n) + (w)) / (w) * w)
typedef int16 v_int;

typedef struct {
    uint c0, c1, c2;
} ranges;

ranges
slice_work(uint n_items, uint n_workers, uint id, uint width) {
    ranges r;
    uint chunk = ALIGN_TO(n_items / n_workers + 1, width);
    r.c0 = id * chunk;
    uint n = min(r.c0 + chunk, n_items) - r.c0;
    r.c1 = r.c0 + n / width * width;
    r.c2 = r.c1 + n % width;
    return r;
}

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

    ranges r = slice_work(N, gs, id, SW);
    v_int scnt = 0;
    int rem = 0;
    for (int i = r.c0; i < r.c1; i += SW) {
        v_int a = VLOAD(i / SW, arr);
        v_int mask = (a % 7) == 0;
        scnt -= mask;
    }
    for (int j = r.c1; j < r.c2; j++) {
        if (arr[j] % 7 == 0) {
            rem++;
        }
    }
    int lcnt = 0;
    for (int i = 0; i < SW; i++) {
        lcnt += scnt[i];
    }
    lcnt = work_group_reduce_add(lcnt);
    if (id == 0) {
        cnt[0] = lcnt;
    }
}
