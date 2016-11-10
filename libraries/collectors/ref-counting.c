// Copyright (C) 2016 Björn Lindqvist
#include <stdlib.h>
#include "quickfit/quickfit.h"
#include "collectors/common.h"
#include "collectors/ref-counting.h"

// Reference counting is a bit faster when using malloc/free over my
// quickfit-allocator. Perhaps because my qf_free_block() function is
// not well optimized.
ref_counting_gc *
rc_init(ptr start, size_t size) {
    ref_counting_gc *me = malloc(sizeof(ref_counting_gc));
    me->size = size;
    me->qf = qf_init(start, size);
    me->decrefs = v_init(16);
    return me;
}

void
rc_free(ref_counting_gc *me) {
    v_free(me->decrefs);
    qf_free(me->qf);
    free(me);
}

bool
rc_can_allot_p(ref_counting_gc *me, size_t size) {
    return qf_can_allot_p(me->qf, size);
}

void
rc_collect(ref_counting_gc *me, vector *roots) {
}

size_t
rc_space_used(ref_counting_gc *me) {
    return me->size - me->qf->free_space;
}

ptr
rc_do_allot(ref_counting_gc *me, int type, size_t size) {
    ptr p = qf_allot_block(me->qf, size);
    P_SET_TYPE(p, type);
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
            qf_free_block(me->qf, p, QF_GET_BLOCK_SIZE(p));
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
