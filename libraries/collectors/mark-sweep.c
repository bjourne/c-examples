// Copyright (C) 2016 Björn Lindqvist
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "collectors/common.h"
#include "collectors/mark-sweep.h"

mark_sweep_gc *
ms_init(size_t size) {
    mark_sweep_gc *me = malloc(sizeof(mark_sweep_gc));
    me->used = 0;
    me->mark_stack = v_init(16);
    me->start = (ptr)malloc(size);
    memset((void *)me->start, 0, size);
    me->size = size;
    me->qf = qf_init(me->start, size);
    return me;
}

void
ms_free(mark_sweep_gc* me) {
    v_free(me->mark_stack);
    qf_free(me->qf);
    free((void *)me->start);
    free(me);
}

static inline void
mark_step(vector *v, ptr p) {
    // Mark the pointer and move it to the gray set.
    if (!P_GET_MARK(p)) {
        P_MARK(p);
        v_add(v, p);
    }
}

void
ms_collect(mark_sweep_gc *me, vector *roots) {
    // Initally, the white set contains all objects, the black and
    // grey sets are empty.
    vector *v = me->mark_stack;
    // First all root object are added to the gray set.
    for (size_t i = 0; i < roots->used; i++) {
        ptr p = roots->array[i];
        if (p) {
            mark_step(v, p);
        }
    }
    me->used = 0;
    while (v->used) {
        // Think of removing the object from the mark stack as moving
        // it from the gray to the black set.
        ptr p = v_remove(v);
        me->used += p_size(p);
        P_FOR_EACH_CHILD(p, {
            mark_step(v, p_child);
        });
    }

    // When control has reached this point, the gray set is empty and
    // the whole heap has been divided into black (marked) and white
    // (condemned) objects.
    qf_clear(me->qf);
    ptr end = me->start + me->size;
    ptr iter = me->start;

    // We should use state bits instead.
    while (iter != end) {
        // Find next unmarked blocks.
        while (P_GET_MARK(iter)) {
            P_UNMARK(iter);
            iter += QF_GET_BLOCK_SIZE(iter);
            if (iter == end) {
                return;
            }
        }
        // Found an unmarked block.
        ptr free_start = iter;
        while (iter != end && !P_GET_MARK(iter)) {
            iter += QF_GET_BLOCK_SIZE(iter);
        }
        size_t free_size = iter - free_start;
        qf_free_block(me->qf, free_start, free_size);
    }
}

bool
ms_can_allot_p(mark_sweep_gc *me, size_t size) {
    return qf_can_allot_p(me->qf, size);
}

ptr
ms_do_allot(mark_sweep_gc *me, int type, size_t size) {
    // Malloc and record address.
    ptr p = qf_allot_block(me->qf, size);
    me->used += size;
    P_SET_TYPE(p, type);
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

static gc_dispatch
table = {
    (gc_func_init)ms_init,
    (gc_func_free)ms_free,
    (gc_func_can_allot_p)ms_can_allot_p,
    (gc_func_collect)ms_collect,
    (gc_func_do_allot)ms_do_allot,
    (gc_func_set_ptr)ms_set_ptr,
    (gc_func_set_ptr)ms_set_new_ptr,
    (gc_func_space_used)ms_space_used
};

gc_dispatch *
ms_get_dispatch_table() {
    return &table;
}
