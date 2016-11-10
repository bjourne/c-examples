#ifndef REF_COUNTING_H
#define REF_COUNTING_H

#include "datatypes/vector.h"

typedef struct {
    size_t size;
    size_t used;
    vector *decrefs;
} ref_counting_gc;

gc_dispatch *rc_get_dispatch_table();


#endif
