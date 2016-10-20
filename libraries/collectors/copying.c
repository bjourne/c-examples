// Copyright (C) 2016 Björn Lindqvist

// A simple garbage collector using bump pointer allocation and semi
// space copying.

// Allocation is done in the active space and when it is full all live
// data is copied to the inactive space. Then the inactive space is
// made the active one and it continues.

// This means that half the memory is always unused, which is very
// wasteful. The upside is that bump pointer allocation is extremely
// fast.
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include "collectors/common.h"
#include "collectors/copying.h"
#include "datatypes/vector.h"

static
space *s_init(size_t size) {
    space *s = malloc(sizeof(space));
    ptr mem = (ptr)malloc(size);
    s->start = mem;
    s->end = mem + size;
    s->here = s->start;
    return s;
}

static
void s_free(space *s) {
    free((void *)s->start);
    free(s);
}

static
ptr s_allot(space *s, size_t n_bytes) {
    assert(s->start <= s->here);
    assert(s->here <= s->end);
    ptr p = s->here;
    s->here += n_bytes;
    return p;
}

// The pointer is copied to the space. Then a forwarding pointer is
// set up from the old location.
static
ptr s_copy_pointer(space *s, ptr p) {
    if (p == 0)
        return p;
    // Check if it has already been forwarded. If so, return its
    // existing forwarding address instead of copying anew.
    ptr header = AT(p);
    if ((header & 1) == 1) {
        return header & ~1;
    }
    size_t n_bytes = p_size(p);

    ptr dst = s_allot(s, n_bytes);
    memcpy((void *)dst, (void *)p, n_bytes);
    AT(p) = dst | 1;
    return dst;
}

static
void s_copy_slots(space *s, ptr *base, size_t n_slots) {
    for (size_t n = 0; n < n_slots; n++) {
        base[n] = s_copy_pointer(s, base[n]);
    }
}

copying_gc *
cg_init(size_t size) {
    assert(size % 2 == 0);
    copying_gc *cg = malloc(sizeof(copying_gc));
    cg->active = s_init(size / 2);
    cg->inactive = s_init(size / 2);
    return cg;
}

void
cg_collect(copying_gc *cg, vector *roots) {
    space *target = cg->inactive;
    s_copy_slots(target, roots->array, roots->used);
    ptr p = target->start;
    while (p < target->here) {
        size_t n_slots = p_slot_count(p);
        s_copy_slots(target, SLOT_P(p, 0), n_slots);
        p += p_size(p);
    }
    assert(p == target->here);
    cg->active->here = cg->active->start;

    space *tmp = cg->active;
    cg->active = cg->inactive;
    cg->inactive = tmp;
}

bool
cg_can_allot_p(copying_gc *cg, size_t n_bytes) {
    return (cg->active->here + n_bytes) <= cg->active->end;
}

ptr
cg_do_allot(copying_gc *cg, size_t n_bytes, uint type) {
    ptr p = s_allot(cg->active, n_bytes);
    AT(p) = type << 1;
    return p;
}

void
cg_free(copying_gc *me) {
    s_free(me->active);
    s_free(me->inactive);
    free(me);
}

size_t
cg_space_used(copying_gc *me) {
    return me->active->here - me->active->start;
}

void
cg_set_ptr(copying_gc *me, ptr *from, ptr to) {
    *from = to;
}

void
cg_set_new_ptr(copying_gc *me, ptr *from, ptr to) {
    *from = to;
}

static gc_dispatch
table = {
    (gc_func_init)cg_init,
    (gc_func_free)cg_free,
    (gc_func_can_allot_p)cg_can_allot_p,
    (gc_func_collect)cg_collect,
    (gc_func_do_allot)cg_do_allot,
    (gc_func_set_ptr)cg_set_ptr,
    (gc_func_set_ptr)cg_set_new_ptr,
    (gc_func_space_used)cg_space_used
};

gc_dispatch *
cg_get_dispatch_table() {
    return &table;
}
