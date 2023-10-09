__kernel void
loop(volatile __global uint *buf) {
    while (buf[0] != 500) {
        buf[1]++;
    }
}

__kernel void
post(volatile __global uint *buf) {
    buf[0] = 500;
}
