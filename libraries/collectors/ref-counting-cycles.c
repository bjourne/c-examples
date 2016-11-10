// Copyright (C) 2016 Björn Lindqvist
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/hashset.h"
#include "collectors/common.h"
#include "collectors/ref-counting.h"
#include "collectors/ref-counting-cycles.h"

static ref_counting_cycles_gc *
rcc_init(ptr start, size_t max_used) {
    ref_counting_cycles_gc *me = malloc(sizeof(ref_counting_cycles_gc));
    me->size = max_used;
    me->used = 0;
    me->blacks = v_init(16);
    me->grays = v_init(16);
    me->whites = v_init(16);
    me->decrefs = v_init(16);
    me->candidates = hs_init();
    return me;
}

static void
rcc_free_p(ref_counting_cycles_gc *me, ptr p, size_t n_bytes) {
    me->used -= n_bytes;
    free((ptr *)p);
}

static void
rcc_mark_gray(ref_counting_cycles_gc *me, ptr p) {
    vector *st = me->grays;
    P_SET_COL(p, COL_GRAY);
    P_FOR_EACH_CHILD(p, {
        P_DEC_RC(p_child);
        size_t col = P_GET_COL(p_child);
        if (col != COL_BLACK && col != COL_GRAY) {
            P_SET_COL(p_child, COL_GRAY);
            v_add(st, p_child);
        }
    });
    while (st->used) {
        p = v_remove(st);
        P_FOR_EACH_CHILD(p, {
            P_DEC_RC(p_child);
            size_t col = P_GET_COL(p_child);
            if (col != COL_BLACK && col != COL_GRAY) {
                P_SET_COL(p_child, COL_GRAY);
                v_add(st, p_child);
            }
        });
    }
}

static void
rcc_mark_candidates(ref_counting_cycles_gc *me) {
    hashset *c = me->candidates;
    HS_FOR_EACH_ITEM(c, {
        if (P_GET_COL(p) == COL_PURPLE) {
            rcc_mark_gray(me, p);
        } else {
            hs_remove_at(c, _i);
            if (P_GET_COL(p) == COL_BLACK && P_GET_RC(p) == 0) {
                rcc_free_p(me, p, p_size(p));
            }
        }
    });
}

static void
rcc_scan_blacks(ref_counting_cycles_gc *me, ptr p) {
    assert(P_GET_COL(p) == COL_BLACK);
    vector *st = me->blacks;
    v_add(st, p);
    while (st->used) {
        p = v_remove(st);
        P_FOR_EACH_CHILD(p, {
            P_INC_RC(p_child);
            if (P_GET_COL(p_child) != COL_BLACK) {
                P_SET_COL(p_child, COL_BLACK);
                v_add(st, p_child);
            }
        });
    }
}

static inline void
rcc_scan_candidate_step(ref_counting_cycles_gc *me, ptr p) {
    // Already processed?
    if (P_GET_COL(p) != COL_GRAY) {
        return;
    }
    if (P_GET_RC(p) > 0) {
        // External ref found, undo trial deletion.
        P_SET_COL(p, COL_BLACK);
        rcc_scan_blacks(me, p);
    } else {
        // Could be garbage.. investigate children.
        P_SET_COL(p, COL_WHITE);
        v_add(me->whites, p);
    }
}

static void
rcc_scan_candidate(ref_counting_cycles_gc *me, ptr p) {
    vector *v = me->whites;
    rcc_scan_candidate_step(me, p);
    while (v->used) {
        p = v_remove(v);
        P_FOR_EACH_CHILD(p, { rcc_scan_candidate_step(me, p_child); });
    }
}

static void
rcc_collect_white(ref_counting_cycles_gc *me, ptr p) {
    hashset *c = me->candidates;
    if (P_GET_COL(p) == COL_WHITE && !hs_in_p(c, p)) {
        P_SET_COL(p, COL_BLACK);
        size_t n_bytes = p_size(p);
        P_FOR_EACH_CHILD(p, { rcc_collect_white(me, p_child); });
        rcc_free_p(me, p, n_bytes);
    }
}

static void
rcc_collect_candidates(ref_counting_cycles_gc *me) {
    hashset *c = me->candidates;
    HS_FOR_EACH_ITEM(c, {
        hs_remove_at(c, _i);
        rcc_collect_white(me, p);
    });
}

void
rcc_collect(ref_counting_cycles_gc *me) {
    rcc_mark_candidates(me);
    hashset *c = me->candidates;
    HS_FOR_EACH_ITEM(c, { rcc_scan_candidate(me, p); });
    rcc_collect_candidates(me);
    assert(c->n_items == 0);
    hs_clear(c);
}

static void
rcc_candidate(ref_counting_cycles_gc *me, ptr p) {
    if (P_GET_COL(p) != COL_PURPLE) {
        size_t t = P_GET_TYPE(p);
        if (TYPE_CONTAINER_P(t)) {
            P_SET_COL(p, COL_PURPLE);
            hs_add(me->candidates, p);
        }
    }
}

static void
rcc_release(ref_counting_cycles_gc *me, ptr p) {
    P_FOR_EACH_CHILD(p, { v_add(me->decrefs, p_child); });
    // The handbook recommends keeping the pointer and waiting for a
    // collection cycle. I don't understand why you shouldn't just
    // free the pointer immediately.
    hs_remove(me->candidates, p);
    rcc_free_p(me, p, p_size(p));
}

static void
rcc_decref(ref_counting_cycles_gc *me, ptr p) {
    vector *st = me->decrefs;
    v_add(st, p);
    while (st->used) {
        p = v_remove(st);
        P_DEC_RC(p);
        if (P_GET_RC(p) == 0) {
            rcc_release(me, p);
        } else {
            rcc_candidate(me, p);
        }
    }
}

static inline void
rcc_addref(ref_counting_cycles_gc *me, ptr p) {
    if (p != 0) {
        P_INC_RC(p);
        P_SET_COL(p, COL_BLACK);
    }
}

void
rcc_free(ref_counting_cycles_gc *me) {
    rcc_collect(me);
    v_free(me->blacks);
    v_free(me->grays);
    v_free(me->whites);
    v_free(me->decrefs);
    hs_free(me->candidates);
    free(me);
}

void
rcc_set_ptr(ref_counting_cycles_gc *me, ptr *from, ptr to) {
    rcc_addref(me, to);
    if (*from != 0) {
        rcc_decref(me, *from);
    }
    *from = to;
}

void
rcc_set_new_ptr(ref_counting_cycles_gc *me, ptr *from, ptr to) {
    rcc_addref(me, to);
    *from = to;
}

static gc_dispatch
table = {
    (gc_func_init)rcc_init,
    (gc_func_free)rcc_free,
    (gc_func_can_allot_p)rc_can_allot_p,
    (gc_func_collect)rcc_collect,
    (gc_func_do_allot)rc_do_allot,
    (gc_func_set_ptr)rcc_set_ptr,
    (gc_func_set_ptr)rcc_set_new_ptr,
    (gc_func_space_used)rc_space_used
};

gc_dispatch *
rcc_get_dispatch_table() {
    return &table;
}
