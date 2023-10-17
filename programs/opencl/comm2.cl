// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Same structure as previously, but now we send messages from the
// consumer back to the producer.
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel int to_consumer __attribute__((depth(64)));
channel int to_producer __attribute__((depth(64)));

__kernel void
consumer(uint n, __global int *arr) {
    for (uint i = 0; i < n; i++) {
        uint r = read_channel_intel(to_consumer);
        write_channel_intel(to_producer, 2 * r);
    }
}

__kernel void
producer(uint n, __global int *arr) {
    uint tot = 0;
    for (uint i = 0; i < n; i++) {
        write_channel_intel(chan, arr[i]);
        tot += read_channel_intel(to_producer);
    }
    arr[0] = tot;
}
