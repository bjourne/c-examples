// Copyright (C) 2016 Björn Lindqvist
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/bitarray.h"
#include "collectors/common.h"
#include "collectors/mark-sweep-bits.h"

mark_sweep_bits_gc *
msb_init(ptr start, size_t size) {
    mark_sweep_bits_gc *me = malloc(sizeof(mark_sweep_bits_gc));
    me->mark_stack = v_init(16);
    me->qf = qf_init(start, size);
    me->ba = ba_init(size / QF_DATA_ALIGNMENT);
    return me;
}

void
msb_free(mark_sweep_bits_gc* me) {
    v_free(me->mark_stack);
    qf_free(me->qf);
    ba_free(me->ba);
    free(me);
}

static inline int
msb_address_to_bit(mark_sweep_bits_gc *me, ptr p) {
    return (p - me->qf->start) / QF_DATA_ALIGNMENT;
}

static inline ptr
msb_bit_to_address(mark_sweep_bits_gc *me, int addr) {
    return me->qf->start + (addr * QF_DATA_ALIGNMENT);
}

static inline void
msb_mark_step(mark_sweep_bits_gc *me, vector *v, ptr p) {
    // Mark the pointer and move it to the gray set.
    int bit_addr = msb_address_to_bit(me, p);
    assert(bit_addr >= 0);
    if (!ba_get_bit(me->ba, bit_addr)) {
        size_t block_size = QF_GET_BLOCK_SIZE(p);
        int n_bits = block_size / QF_DATA_ALIGNMENT;
        ba_set_bit_range(me->ba, bit_addr, n_bits);
        v_add(v, p);
    }
}

void
msb_collect(mark_sweep_bits_gc *me, vector *roots) {
    ba_clear(me->ba);
    vector *v = me->mark_stack;
    for (size_t i = 0; i < roots->used; i++) {
        ptr p = roots->array[i];
        if (p) {
            msb_mark_step(me, v, p);
        }
    }
    while (v->used) {
        ptr p = v_remove(v);
        P_FOR_EACH_CHILD(p, { msb_mark_step(me, v, p_child); });
    }
    qf_clear(me->qf);
    BA_EACH_UNSET_RANGE(me->ba, {
        ptr free_start = msb_bit_to_address(me, addr);
        size_t free_size = size * QF_DATA_ALIGNMENT;
        qf_free_block(me->qf, free_start, free_size);
    });
}

bool
msb_can_allot_p(mark_sweep_bits_gc *me, size_t size) {
    return qf_can_allot_p(me->qf, size);
}

ptr
msb_do_allot(mark_sweep_bits_gc *me, int type, size_t size) {
    // Allot and record address.
    ptr p = qf_allot_block(me->qf, size);
    P_SET_TYPE(p, type);
    return p;
}

size_t
msb_space_used(mark_sweep_bits_gc *me) {
    return qf_space_used(me->qf);
}

void
msb_set_ptr(mark_sweep_bits_gc *ms, ptr *from, ptr to) {
    *from = to;
}

void
msb_set_new_ptr(mark_sweep_bits_gc *ms, ptr *from, ptr to) {
    *from = to;
}

static gc_dispatch
table = {
    (gc_func_init)msb_init,
    (gc_func_free)msb_free,
    (gc_func_can_allot_p)msb_can_allot_p,
    (gc_func_collect)msb_collect,
    (gc_func_do_allot)msb_do_allot,
    (gc_func_set_ptr)msb_set_ptr,
    (gc_func_set_ptr)msb_set_new_ptr,
    (gc_func_space_used)msb_space_used
};

gc_dispatch *
msb_get_dispatch_table() {
    return &table;
}
