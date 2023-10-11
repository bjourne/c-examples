// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>

__kernel void
add_reduce(const uint N, const __global int *A, __global int *ans) {

    uint id = get_local_id(0);
    uint n_els = N / get_local_size(0);

    uint ofs0 = n_els * id;
    uint ofs1 = ofs0 + n_els;

    int sum = 0;
    for (uint i = ofs0; i < ofs1; i++) {
        sum += A[i];
    }
    sum = work_group_reduce_add(sum);
    if (id == 0) {
        ans[0] = sum;
    }
}
