#ifndef REF_COUNTING_H
#define REF_COUNTING_H

#include "datatypes/vector.h"

typedef struct {
    size_t size;
    size_t used;
    vector *decrefs;
} ref_counting_gc;

gc_dispatch *rc_get_dispatch_table();

bool rc_can_allot_p(ref_counting_gc *me, size_t n_bytes);
ptr rc_do_allot(ref_counting_gc *me, int type, size_t n_bytes);
size_t rc_space_used(ref_counting_gc *me);

#endif
