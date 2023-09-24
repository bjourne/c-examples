// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdlib.h>
#include "gens_set.h"

gens_set *
gens_set_init(size_t size) {
    gens_set *me = malloc(sizeof(gens_set));
    me->array = calloc(size, sizeof(uint32_t));
    me->size = size;
    me->used = 0;
    me->seq = 1;
    return me;
}

void
gens_set_free(gens_set *me) {
    free(me->array);
    free(me);
}

extern inline bool
gens_set_contains(gens_set *me, uint32_t k);

extern inline bool
gens_set_add(gens_set *me, uint32_t k);

extern inline bool
gens_set_remove(gens_set *me, uint32_t k);

extern inline void
gens_set_clear(gens_set *me);
