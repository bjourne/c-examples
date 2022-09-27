__kernel void
producer(write_only pipe uint __attribute__((blocking)) c0) {
    for (uint i = 0; i < 10; i++) {
        write_pipe(c0, &i);
    }
}
