#pragma OPENCL EXTENSION cl_intel_channels : enable

channel int chan __attribute__((depth(64)));

__kernel void
consumer(uint n, __global int *ret) {
    ret[0] = 123;
}

__kernel void
producer(uint n, __global int *arr) {
    for (uint i = 0; i < n; i++) {
        write_channel_intel(chan, arr[i]);
    }
}
