// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// My version of www.pcg-random.org
#ifndef RANDOM_H
#define RANDOM_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint64_t state;
    // Inc must always be odd.
    uint64_t inc;
    // Initialized
    bool initialized;
} rnd_pcg32;

rnd_pcg32 rnd_pcg32_get_state();

void rnd_pcg32_seed(uint64_t init_state, uint64_t init_seq);

uint32_t rnd_pcg32_rand();
// Bias-avoiding, so it's better than rnd_pcg32_rand() % N.
uint32_t rnd_pcg32_rand_range(uint32_t lim);

// Random double in the interval [0, 1)
double rnd_pcg32_rand_double_0_to_1();

// Generate a sequence of random numbers 0 <= r < lim.
void
rnd_pcg32_rand_range_fill(uint32_t *mem, uint32_t lim, size_t n);

void
rnd_pcg32_rand_uniform_fill_float(float *mem, size_t n);


#endif
