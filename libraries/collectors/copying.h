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
} copying_gc;

// Init, free
copying_gc *cg_init(size_t size);
void cg_free(copying_gc *me);

// Allocation
bool cg_can_allot_p(copying_gc *me, size_t n_bytes);
void cg_collect(copying_gc *me, vector *roots);
ptr cg_do_allot(copying_gc *me, size_t n_bytes);

// To facilitate barriers and refcounting.
void cg_set_ptr(copying_gc *me, ptr *from, ptr to);
void cg_set_new_ptr(copying_gc *me, ptr *from, ptr to);

// Stats
size_t cg_space_used(copying_gc *me);

// Interface support
gc_dispatch *cg_get_dispatch_table();

#endif
