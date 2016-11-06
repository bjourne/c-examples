#ifndef ONESIE_H
#define ONESIE_H

// This is a dirt-simple allocator

#include <stdbool.h>
#include "datatypes/vector.h"

typedef struct {
    vector* free_blocks;
    ptr region;
} onesie;

onesie *os_init(size_t n, size_t size);
void os_free(onesie *os);

bool os_can_allot_p(onesie *os);
ptr os_allot_block(onesie *os);
void os_free_block(onesie *os, ptr block);

#endif
