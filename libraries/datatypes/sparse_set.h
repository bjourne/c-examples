// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef SPARSE_SET_H
#define SPARSE_SET_H

#include <stdbool.h>

// See https://research.swtch.com/sparse

typedef struct {
    int *dense;
    int *sparse;
    int size;
    int used;
} sparse_set;

sparse_set *sparse_set_init(int n);
void sparse_set_free(sparse_set *me);

inline bool
sparse_set_add(sparse_set *me, int k) {
    int i = me->sparse[k];
    if (i < me->used && me->dense[i] == k) {
        return false;
    }
    me->dense[me->used] = k;
    me->sparse[k] = me->used;
    me->used++;
    return true;
}

inline bool
sparse_set_remove(sparse_set *me, int k) {
    int i = me->sparse[k];
    if (i < me->used && me->dense[i] == k) {
        int y = me->dense[me->used - 1];
        me->dense[i] = y;
        me->sparse[y] = i;
        me->used--;
        return true;
    }
    return false;
}

inline bool
sparse_set_contains(sparse_set *me, int k) {
    int i = me->sparse[k];
    return i < me->used && me->dense[i] == k;
}

inline void
sparse_set_clear(sparse_set *me) {
    me->used = 0;
}


#endif
