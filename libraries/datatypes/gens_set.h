// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef GENS_SET_H
#define GENS_SET_H

#include <stdbool.h>
#include <stdint.h>

// See https://quasilyte.dev/blog/post/gen-map/
typedef struct {
    uint32_t *array;
    uint32_t seq;
    size_t size;
    size_t used;
} gens_set;

gens_set *gens_set_init(size_t size);
void gens_set_free(gens_set *me);

inline bool
gens_set_contains(gens_set *me, uint32_t k) {
    return me->array[k] == me->seq;
}

inline bool
gens_set_add(gens_set *me, uint32_t k) {
    uint32_t at = me->seq;
    uint32_t old = me->array[k];
    me->array[k] = at;
    return old != at;
}

inline bool
gens_set_remove(gens_set *me, uint32_t k) {
    bool old = me->array[k];
    me->array[k] = 0;
    return old == me->seq;
}

inline void
gens_set_clear(gens_set *me) {
    me->seq++;
}

#endif
