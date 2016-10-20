// Copyright (C) 2016 Björn Lindqvist
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "collectors/common.h"
#include "collectors/mark-sweep.h"

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
    if (!P_GET_MARK(p)) {
        AT(p) |= 1;
        v_add(v, p);
    }
}

void
ms_collect(mark_sweep_gc *ms) {
    // Initally, the white set contains all objects, the black and
    // grey sets are empty.
    vector *v = ms->mark_stack;
    // First all root object are added to the gray set.
    for (size_t i = 0; i < ms->roots->used; i++) {
        ptr p = ms->roots->array[i];
        if (p) {
            mark_step(v, p);
        }
    }
    while (v->used) {
        // Think of removing the object from the mark stack as moving
        // it from the gray to the black set.
        ptr p = v_remove(v);
        P_FOR_EACH_CHILD(p, {
            mark_step(v, p_child);
        });
    }
    // When control has reached this point, the gray set is empty and
    // the whole heap has been divided into black (marked) and white
    // (condemned) objects.

    // Sweep phase
    allot_link **scan = &ms->allots;
    while (*scan) {
        allot_link *at = *scan;
        ptr p = (ptr)at + LINK_SIZE;
        if (!P_GET_MARK(p)) {
            ms->used -= p_size(p);
            *scan = at->next;
            free(at);
        } else {
            AT(p) &= ~1;
            scan = &at->next;
        }
    }
}

bool
ms_can_allot_p(mark_sweep_gc *ms, size_t n_bytes) {
    return (ms->used + n_bytes) <= ms->size;
}

ptr
ms_do_allot(mark_sweep_gc *ms, size_t n_bytes, uint type) {
    // Malloc and record address.
    size_t n_bytes_req = n_bytes + LINK_SIZE;
    allot_link* link = malloc(n_bytes_req);
    link->next = ms->allots;
    ms->allots = link;
    ms->used += n_bytes;
    ptr p = (ptr)link + LINK_SIZE;
    AT(p) = type << 1;
    return p;
}

size_t
ms_space_used(mark_sweep_gc *ms) {
    return ms->used;
}

void
ms_set_ptr(mark_sweep_gc *ms, ptr *from, ptr to) {
    *from = to;
}

void
ms_set_new_ptr(mark_sweep_gc *ms, ptr *from, ptr to) {
    *from = to;
}
