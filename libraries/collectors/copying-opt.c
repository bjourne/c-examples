// Copyright (C) 2016 Björn Lindqvist
#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "collectors/common.h"
#include "collectors/copying.h"

// This is the performance optimized version of copying.c

static
ptr s_allot(space *s, size_t n_bytes) {
    assert(s->start <= s->here);
    assert(s->here <= s->end);
    ptr p = s->here;
    s->here += n_bytes;
    return p;
}

//__attribute__ ((noinline)) <- why did I want this?
void fast_memcpy(ptr *dst, ptr* src, size_t n) {
    if (n == 16) {
        *dst++ = *src++;
        *dst++ = *src++;
    } else {
        #if _WIN32
        memcpy(dst, src, n);
        #else
        size_t n_ptrs = n / sizeof(ptr);
        asm volatile("cld\n\t"
                     "rep ; movsq"
                     : "=D" (dst), "=S" (src)
                     : "c" (n_ptrs), "D" (dst), "S" (src)
                     : "memory");
        #endif
    }
}

static inline
ptr s_copy_pointer(space *target, ptr p) {
    ptr header = AT(p);
    if ((header & 1) == 1) {
        return header & ~1;
    }

    size_t n_bytes = NPTRS(2);
    if ((header >> 1) == TYPE_ARRAY) {
        size_t n_els = *SLOT_P(*SLOT_P(p, 0), 0);
        n_bytes = NPTRS(2 + n_els);
    }

    ptr dst = s_allot(target, n_bytes);
    fast_memcpy((ptr *)dst, (ptr *)p, n_bytes);
    AT(p) = dst | 1;
    return dst;
}

static
void s_copy_slots(space *target, ptr *base, ptr *end) {
    while (base < end) {
        ptr p = *base;
        if (p != 0) {
            *base = s_copy_pointer(target, p);
        }
        base++;
    }
}

void
cg_collect_optimized(copying_gc *me, vector *roots) {
    space *target = me->inactive;
    s_copy_slots(target, roots->array, roots->array + roots->used);
    ptr p = target->start;
    while (p < target->here) {
        size_t t = P_GET_TYPE(p);
        switch (t) {
        case TYPE_INT:
        case TYPE_FLOAT: {
            p += NPTRS(2);
            break;
        }
        case TYPE_WRAPPER: {
            ptr *slot0 = SLOT_P(p, 0);
            if (*slot0 != 0) {
                *slot0 = s_copy_pointer(target, *slot0);
            }
            p += NPTRS(2);
            break;
        }
        case TYPE_ARRAY: {
            ptr *slot0 = SLOT_P(p, 0);
            size_t n_els = *SLOT_P(*slot0, 0);
            p += NPTRS(2 + n_els);
            s_copy_slots(target, slot0, (ptr *)p);
            break;
        }
        }
    }
    me->active->here = me->active->start;

    space *tmp = me->active;
    me->active = me->inactive;
    me->inactive = tmp;
}

static gc_dispatch
table = {
    (gc_func_init)cg_init,
    (gc_func_free)cg_free,
    (gc_func_can_allot_p)cg_can_allot_p,
    (gc_func_collect)cg_collect_optimized,
    (gc_func_do_allot)cg_do_allot,
    (gc_func_set_ptr)cg_set_ptr,
    (gc_func_set_ptr)cg_set_new_ptr,
    (gc_func_space_used)cg_space_used
};

gc_dispatch *
cg_get_dispatch_table_optimized() {
    return &table;
}
