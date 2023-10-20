// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
__kernel void
prefix_sum(const ulong N, const __global float *arr, __global float *pf) {
    uint id = get_local_id(0);
    uint gid = get_group_id(0);

    uint ofs = 0;
    float max = 0;
    do {
        uint idx = ofs + id;
        if (idx < N) {
            float v = arr[idx];
            float s = work_group_scan_exclusive_add(v);
            pf[idx] = s + max;
            max += work_group_broadcast(s + v, get_local_size(0) - 1);
        }
        ofs += get_local_size(0);
    } while (ofs < N);
}
