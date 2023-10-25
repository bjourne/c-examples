// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Count the number of elements divisible by 7, with and without SIMD.
__kernel void
count_divisible(const uint N,
                const __global uint * restrict arr,
                __global ulong * restrict cnt) {

    uint gs = get_local_size(0);
    uint id = get_local_id(0);


    uint chunk = N / gs + 1;

    uint i0 = id * chunk;
    uint i1 = min(i0 + chunk, N);
    /* uint n = i1 - i0; */

    /* uint n_rem = n % 4; */
    /* uint n_chunks = n / 4; */

    /* for (uint i = 0; i < n_chunks; i++) { */
    /*     for  */
    /* } */

    uint lcnt = 0;
    for (uint i = i0; i < i1; i++) {
        if (arr[i] % 7 == 0) {
            lcnt++;
        }
    }
    lcnt = work_group_reduce_add(lcnt);
    if (id == 0) {
        cnt[0] = lcnt;
    }
}

// On my machine int16 is slighty faster than int8.
#define SW  16
#define ALIGN(n)    ((n) + SW) / SW * SW;
#define VLOAD(ofs, arr) vload16((ofs), (arr))
typedef int16 v_int;

__kernel void
count_divisible_simd(const int N,
                     const __global int * restrict arr,
                     __global ulong * restrict cnt) {
    int gs = get_local_size(0);
    int id = get_local_id(0);

    int chunk = ALIGN(N / gs + 1);
    int c0 = id * chunk;
    int n = min(c0 + chunk, N) - c0;

    int n_rem = n % SW;
    int c1 = c0 + n / SW * SW;
    int c2 = c1 + n % SW;

    v_int scnt = 0;
    int rem = 0;
    for (int i = c0; i < c1; i += SW) {
        v_int a = VLOAD(i / SW, arr);
        v_int mask = (a % 7) == 0;
        scnt -= mask;
    }
    for (int j = c1; j < c2; j++) {
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
