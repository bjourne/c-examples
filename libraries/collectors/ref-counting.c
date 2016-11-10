// Copyright (C) 2016 Björn Lindqvist
#include <stdlib.h>
#include "collectors/common.h"
#include "collectors/ref-counting.h"

ref_counting_gc *
rc_init(size_t max_used) {
    ref_counting_gc *me = malloc(sizeof(ref_counting_gc));
    me->size = max_used;
    me->used = 0;
    me->decrefs = v_init(16);
    return me;
}

void
rc_free(ref_counting_gc *me) {
    v_free(me->decrefs);
    free(me);
}

bool
rc_can_allot_p(ref_counting_gc *me, size_t n_bytes) {
    return (me->used + n_bytes) <= me->size;
}

void
rc_collect(ref_counting_gc *me, vector *roots) {
}

size_t
rc_space_used(ref_counting_gc *me) {
    return me->used;
}

static ptr
rc_do_allot(ref_counting_gc *me, int type, size_t n_bytes) {
    me->used += n_bytes;
    ptr p = (ptr)malloc(n_bytes);
    AT(p) = type << 1;
    return p;
}

static void
rc_decref(ref_counting_gc *me, ptr p) {
    if (p == 0) {
        return;
    }
    vector *v = me->decrefs;
    v_add(v, p);
    while (v->used) {
        p = v_remove(v);
        P_DEC_RC(p);
        if (P_GET_RC(p) == 0) {
            P_FOR_EACH_CHILD(p, { v_add(v, p_child); });
            me->used -= p_size(p);
            free((ptr *)p);
        }
    }
}

static inline void
rc_addref(ref_counting_gc *me, ptr p) {
    if (p != 0) {
        P_INC_RC(p);
    }
}

void
rc_set_ptr(ref_counting_gc *me, ptr *from, ptr to) {
    rc_decref(me, *from);
    rc_addref(me, to);
    *from = to;
}

void
rc_set_new_ptr(ref_counting_gc *me, ptr *from, ptr to) {
    rc_addref(me, to);
    *from = to;
}

static gc_dispatch
table = {
    (gc_func_init)rc_init,
    (gc_func_free)rc_free,
    (gc_func_can_allot_p)rc_can_allot_p,
    (gc_func_collect)rc_collect,
    (gc_func_do_allot)rc_do_allot,
    (gc_func_set_ptr)rc_set_ptr,
    (gc_func_set_ptr)rc_set_new_ptr,
    (gc_func_space_used)rc_space_used
};

gc_dispatch *
rc_get_dispatch_table() {
    return &table;
}
