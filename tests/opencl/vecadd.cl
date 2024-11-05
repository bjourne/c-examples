// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
__kernel void vecadd(
    __global const float * restrict a,
    __global const float * restrict b,
    __global float * restrict c
) {
    size_t i = get_global_id(0);
    float av = a[i], bv = b[i];
    c[i] = av + bv;
}
