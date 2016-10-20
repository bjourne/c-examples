#ifndef COLLECTORS_COMMON_H
#define COLLECTORS_COMMON_H

#include <stdlib.h>
#include "datatypes/common.h"

// Object header: | 59: unused | 4: type | 1: mark
#define P_GET_MARK(p) (AT(p) & 1)
#define P_GET_TYPE(p) AT(p) >> 1

// Object types
#define TYPE_INT 1
#define TYPE_FLOAT 2
#define TYPE_WRAPPER 3
#define TYPE_ARRAY 4

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
void p_print_slots(size_t ind, ptr *base, size_t n);

#endif
