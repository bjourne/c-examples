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

void rnd_pcg32_seed(uint64_t init_state, uint64_t init_seq);

uint32_t rnd_pcg32_rand();
// Bias-avoiding, so it's better than rnd_pcg32_rand() % N.
uint32_t rnd_pcg32_rand_range(uint32_t lim);


#endif
