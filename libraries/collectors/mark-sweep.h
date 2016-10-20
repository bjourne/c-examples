#ifndef MARK_SWEEP_H
#define MARK_SWEEP_H

#include <stdbool.h>
#include "collectors/common.h"
#include "datatypes/vector.h"

// Object header: | 59: unused | 4: type | 1: mark

typedef struct _allot_link {
    struct _allot_link *next;
} allot_link;

typedef struct {
    size_t size;
    size_t used;
    allot_link *allots;
    vector *mark_stack;
} mark_sweep_gc;

// Init, free
mark_sweep_gc *ms_init(size_t max_used);
void ms_free(mark_sweep_gc *ms);

// Allocation
bool ms_can_allot_p(mark_sweep_gc *me, size_t n_bytes);
void ms_collect(mark_sweep_gc *me, vector *roots);
ptr ms_do_allot(mark_sweep_gc *me, size_t n_bytes, uint type);

// To facilitate barriers and refcounting.
void ms_set_ptr(mark_sweep_gc *me, ptr *from, ptr to);
void ms_set_new_ptr(mark_sweep_gc *me, ptr *from, ptr to);

// Stats
size_t ms_space_used(mark_sweep_gc *me);

// Interface support
gc_dispatch *ms_get_dispatch_table();


#endif
