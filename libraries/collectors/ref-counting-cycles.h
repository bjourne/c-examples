#ifndef REF_COUNTING_CYCLES_H
#define REF_COUNTING_CYCLES_H

#include "datatypes/hashset.h"
#include "datatypes/vector.h"

typedef struct {
    size_t size;
    size_t used;
    vector *blacks;
    vector *grays;
    vector *whites;
    vector *decrefs;
    hashset *candidates;
} ref_counting_cycles_gc;

ref_counting_cycles_gc *rcc_init(size_t max_used);
gc_dispatch *rcc_get_dispatch_table();



#endif
