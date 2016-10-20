// Copyright (C) 2016 Björn Lindqvist
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/vector.h"
#include "mark-sweep.h"

#define LINK_SIZE sizeof(allot_link)

mark_sweep_gc *
ms_init(size_t max_used, vector *roots) {
    mark_sweep_gc *ms = malloc(sizeof(mark_sweep_gc));
    ms->size = max_used;
    ms->used = 0;
    ms->allots = NULL;
    ms->roots = roots;
    ms->mark_stack = v_init(16);
    return ms;
}

void
ms_free(mark_sweep_gc* ms) {
    allot_link *at = ms->allots;
    while (at) {
        allot_link *next = at->next;
        free(at);
        at = next;
    }
    v_free(ms->mark_stack);
    free(ms);
}

static inline void
mark_step(vector *v, ptr p) {
    // Mark the pointer and move it to the gray set.
    if (!MS_GET_MARK(p)) {
        AT(p) |= 1;
        v_add(v, p);
    }
}
