// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <time.h>
#include "random.h"

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL, false }

static rnd_pcg32 glob = PCG32_INITIALIZER;

// A uniformly distributed 32 bit random number.
static uint32_t
rnd_pcg32_rand_int() {
    uint64_t prev_state = glob.state;
    glob.state = prev_state * 6364136223846793005ULL + glob.inc;
    uint32_t xor_shifted = ((prev_state >> 18U) ^ prev_state) >> 27U;
    uint32_t rot = prev_state >> 59U;
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

void
rnd_pcg32_seed(uint64_t init_state, uint64_t init_seq) {
    glob.state = 0U;
    glob.inc = (init_seq << 1U) | 1U;
    rnd_pcg32_rand_int();
    glob.state += init_state;
    rnd_pcg32_rand_int();
}

uint32_t
rnd_pcg32_rand() {
    if (!glob.initialized) {
        rnd_pcg32_seed(time(NULL), (uint64_t)&rnd_pcg32_rand);
        glob.initialized = true;
    }
    return rnd_pcg32_rand_int();
}

uint32_t
rnd_pcg32_rand_range(uint32_t lim) {
    uint32_t thresh = -lim % lim;
    while (true) {
        uint32_t r = rnd_pcg32_rand();
        if (r >= thresh) {
            return r % lim;
        }
    }
}
