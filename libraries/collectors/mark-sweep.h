#ifndef MARK_SWEEP_H
#define MARK_SWEEP_H

#include <stdbool.h>
#include "datatypes/vector.h"
#include "quickfit/quickfit.h"
#include "collectors/common.h"

typedef struct {
    vector *mark_stack;
    quick_fit *qf;
} mark_sweep_gc;

// Init, free
mark_sweep_gc *ms_init(ptr start, size_t size);
void ms_free(mark_sweep_gc *ms);

// Allocation
bool ms_can_allot_p(mark_sweep_gc *me, size_t size);
void ms_collect(mark_sweep_gc *me, vector *roots);
ptr ms_do_allot(mark_sweep_gc *me, int type, size_t size);

// To facilitate barriers and refcounting.
void ms_set_ptr(mark_sweep_gc *me, ptr *from, ptr to);
void ms_set_new_ptr(mark_sweep_gc *me, ptr *from, ptr to);

// Stats
size_t ms_space_used(mark_sweep_gc *me);

// Interface support
gc_dispatch *ms_get_dispatch_table();


#endif
