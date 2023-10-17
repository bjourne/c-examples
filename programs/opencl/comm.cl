// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel int chan __attribute__((depth(64)));

__kernel void
consumer(uint n, __global int *arr) {
    uint r = 0;
    for (uint i = 0; i < n; i++) {
        r += read_channel_intel(chan);
    }
    arr[0] = r;
}

__kernel void
producer(uint n, __global int *arr) {
    for (uint i = 0; i < n; i++) {
        write_channel_intel(chan, arr[i]);
    }
}
