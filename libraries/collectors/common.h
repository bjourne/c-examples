#ifndef COLLECTORS_COMMON_H
#define COLLECTORS_COMMON_H

#include <stdbool.h>
#include <stdlib.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"
#include "datatypes/vector.h"

// New object header:
//
//      32: block size 24: ref count 3: color 4: type 1: mark/forward
//
// It only fits 64bit systems. Probably better with a completely
// different model for 32 bit ones.
#define P_GET_MARK(p)       (AT(p) & 1)
#define P_UNMARK(p)         AT(p) &= ~1
#define P_MARK(p)           AT(p) |= 1

#define P_GET_TYPE(p)       P_GET(p, 1, 4)
#define P_SET_TYPE(p, t)    AT(p) = BF_MERGE(AT(p), t, 1, 4)

#define P_GET_RC(p)         BF_GET(AT(p), 8, 24)
#define P_SET_RC(p, n)      AT(p) = BF_MERGE(AT(p), n, 8, 24)

#define P_DEC_RC(p)         AT(p) = AT(p) - (1L << 8)
#define P_INC_RC(p)         AT(p) = AT(p) + (1L << 8)

#define P_GET_COL(p)        BF_GET(AT(p), 5, 3)
#define P_SET_COL(p, c)     AT(p) = BF_MERGE(AT(p), c, 5, 3)

#define TYPE_CONTAINER_P(t) (t == TYPE_ARRAY || t == TYPE_WRAPPER)

// Object types
#define TYPE_INT        1
#define TYPE_FLOAT      2
#define TYPE_WRAPPER    3
#define TYPE_ARRAY      4

// Object colors, used for ref counting cycles

// Cant be in a cycle.
#define COL_BLACK   0
// Candidate.
#define COL_PURPLE  1
// Possibly the root of a garbage cycle.
#define COL_GRAY 2
// Looks like garbage...
#define COL_WHITE 3


// Utility macros
#define P_FOR_EACH_CHILD(p, body)                               \
    for (size_t _n = p_slot_count(p), _i = 0; _i < _n; _i++) {  \
        ptr p_child = *SLOT_P(p, _i);                           \
        if (p_child) { body }                                   \
    }

#define NPTRS(n)  ((n) * sizeof(ptr))

// Takes an address to an object and outputs a pointer to the given
// slot in that object. It's used for reading and writing slots.
#define SLOT_P(p, n) (ptr *)(p + NPTRS(n + 1))

size_t p_size(ptr p);
size_t p_slot_count(ptr p);
void p_print_slots(int ind, ptr *base, size_t n);


// This is the protocol that any collector must implement.
typedef void *(*gc_func_init)(size_t max_used);
typedef void (*gc_func_free)(void *me);

typedef bool (*gc_func_can_allot_p)(void *me, size_t n_bytes);
typedef void (*gc_func_collect)(void *me, vector *roots);
typedef ptr (*gc_func_do_allot)(void *me, int type, size_t n_bytes);

typedef void (*gc_func_set_ptr)(void *me, ptr *from, ptr to);
typedef void (*gc_func_set_new_ptr)(void *me, ptr *from, ptr to);

typedef size_t (*gc_func_space_used)(void *me);

typedef struct {
    gc_func_init init;
    gc_func_free free;

    gc_func_can_allot_p can_allot_p;
    gc_func_collect collect;
    gc_func_do_allot do_allot;

    gc_func_set_ptr set_ptr;
    gc_func_set_new_ptr set_new_ptr;

    gc_func_space_used space_used;
} gc_dispatch;


#endif
