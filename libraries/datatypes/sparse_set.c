// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdlib.h>
#include "sparse_set.h"

sparse_set *
sparse_set_init(int size) {
    sparse_set *me = malloc(sizeof(sparse_set));
    me->dense = malloc(sizeof(int) * (size_t)size);
    me->sparse = malloc(sizeof(int) * (size_t)size);
    me->size = size;
    me->used = 0;
    return me;
}

void
sparse_set_free(sparse_set *me) {
    free(me->dense);
    free(me->sparse);
    free(me);
}

extern inline bool
sparse_set_add(sparse_set *me, int k);

extern inline bool
sparse_set_remove(sparse_set *me, int k);

extern inline bool
sparse_set_contains(sparse_set *me, int k);

extern inline void
sparse_set_clear(sparse_set *me);
