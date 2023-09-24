// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef SPARSE_SET_H
#define SPARSE_SET_H

#include <stdbool.h>
#include <stdint.h>

// See https://research.swtch.com/sparse

typedef struct {
    uint32_t *dense;
    uint32_t *sparse;
    size_t size;
    size_t used;
} sparse_set;

sparse_set *sparse_set_init(size_t n);
void sparse_set_free(sparse_set *me);

inline bool
sparse_set_add(sparse_set *me, uint32_t k) {
    uint32_t i = me->sparse[k];
    if (i < me->used && me->dense[i] == k) {
        return false;
    }
    me->dense[me->used] = k;
    me->sparse[k] = me->used;
    me->used++;
    return true;
}

inline bool
sparse_set_remove(sparse_set *me, uint32_t k) {
    uint32_t i = me->sparse[k];
    if (i < me->used && me->dense[i] == k) {
        uint32_t y = me->dense[me->used - 1];
        me->dense[i] = y;
        me->sparse[y] = i;
        me->used--;
        return true;
    }
    return false;
}

inline bool
sparse_set_contains(sparse_set *me, uint32_t k) {
    uint32_t i = me->sparse[k];
    return i < me->used && me->dense[i] == k;
}

inline void
sparse_set_clear(sparse_set *me) {
    me->used = 0;
}


#endif
