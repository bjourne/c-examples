// Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#ifndef HASHSET_H
#define HASHSET_H

#include <stdbool.h>
#include "common.h"

// Expand if the fill factor is greater than this.
#define HS_MAX_FILL 0.50
#define HS_INITIAL_CAPACITY 32

#define HS_PRIME_1 73
#define HS_PRIME_2 5009

#define HS_FIRST_KEY(hs, item) ((HS_PRIME_1 * item) & hs->mask)
#define HS_NEXT_KEY(hs, i)  (i + HS_PRIME_2) & hs->mask

#define HS_FOR_EACH_ITEM(hs, body)                          \
    for (size_t _n = hs->capacity, _i = 0; _i < _n; _i++) { \
        ptr p = hs->array[_i];                              \
        if (p > 1) { body }                                 \
    }

typedef struct {
    size_t mask;
    size_t capacity;
    size_t *array;
    size_t n_used;
    size_t n_items;
} hashset;

hashset *hs_init();
void hs_free(hashset *hs);
bool hs_add(hashset *hs, size_t item);
void hs_remove_at(hashset *hs, size_t i);
bool hs_remove(hashset *hs, size_t item);
bool hs_in_p(hashset *hs, size_t item);
void hs_clear(hashset *hs);

#endif
