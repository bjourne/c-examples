#ifndef MARK_SWEEP_H
#define MARK_SWEEP_H

#include <stdbool.h>
#include "datatypes/vector.h"

// Object header: | 59: unused | 4: type | 1: mark

typedef struct _allot_link {
    struct _allot_link *next;
} allot_link;

typedef struct {
    size_t size;
    size_t used;
    allot_link *allots;
    vector *roots;
    vector *mark_stack;
} mark_sweep_gc;

mark_sweep_gc *ms_init(size_t max_used, vector *roots);
void ms_free(mark_sweep_gc* ms);

// Allocation
bool ms_can_allot_p(mark_sweep_gc *ms, size_t n_bytes);
void ms_collect(mark_sweep_gc *ms);
ptr ms_do_allot(mark_sweep_gc *ms, size_t n_bytes, uint type);

// To facilitate barriers and refcounting.
void ms_set_ptr(mark_sweep_gc *ms, ptr *from, ptr to);
void ms_set_new_ptr(mark_sweep_gc *ms, ptr *from, ptr to);

// Stats
size_t ms_space_used(mark_sweep_gc *ms);


#endif
