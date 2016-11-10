#ifndef REF_COUNTING_CYCLES_H
#define REF_COUNTING_CYCLES_H

#include "datatypes/hashset.h"
#include "datatypes/vector.h"

typedef struct {
    quick_fit *qf;
    vector *blacks;
    vector *grays;
    vector *whites;
    vector *decrefs;
    hashset *candidates;
} ref_counting_cycles_gc;

gc_dispatch *rcc_get_dispatch_table();



#endif
