#ifndef COLLECTORS_COPYING_H
#define COLLECTORS_COPYING_H

#include <stdbool.h>
#include "datatypes/vector.h"

typedef struct {
    ptr start;
    ptr end;
    ptr here;
} space;

typedef struct {
    space *active;
    space *inactive;
    vector *roots;
} copying_gc;

// Init, free
copying_gc *cg_init(ptr size, vector *roots);
void cg_free(copying_gc *me);

// Allocation
bool cg_can_allot_p(copying_gc *me, size_t n_bytes);
void cg_collect(copying_gc *me);
ptr cg_do_allot(copying_gc *me, size_t n_bytes, uint type);

// To facilitate barriers and refcounting.
void cg_set_ptr(copying_gc *me, ptr *from, ptr to);
void cg_set_new_ptr(copying_gc *me, ptr *from, ptr to);

// Stats
size_t cg_space_used(copying_gc *me);

#endif
