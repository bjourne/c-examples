#ifndef MARK_SWEEP_BITS_H
#define MARK_SWEEP_BITS_H

#include <stdbool.h>
#include "datatypes/bitarray.h"
#include "datatypes/vector.h"
#include "quickfit/quickfit.h"
#include "collectors/common.h"

// The reason this is a separate collector is because it appears that
// using in-object mark bits can be faster.

typedef struct {
    vector *mark_stack;
    quick_fit *qf;
    bitarray *ba;
} mark_sweep_bits_gc;

// Init, free
mark_sweep_bits_gc *msb_init(ptr start, size_t size);
void msb_free(mark_sweep_bits_gc *ms);

// Allocation
bool msb_can_allot_p(mark_sweep_bits_gc *me, size_t size);
void msb_collect(mark_sweep_bits_gc *me, vector *roots);
ptr msb_do_allot(mark_sweep_bits_gc *me, int type, size_t size);


// Interface support
gc_dispatch *msb_get_dispatch_table();


#endif
