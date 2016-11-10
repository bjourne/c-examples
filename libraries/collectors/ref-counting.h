#ifndef REF_COUNTING_H
#define REF_COUNTING_H

#include <stdbool.h>
#include "datatypes/vector.h"
#include "quickfit/quickfit.h"
#include "collectors/common.h"

typedef struct {
    size_t size;
    quick_fit *qf;
    vector *decrefs;

} ref_counting_gc;

gc_dispatch *rc_get_dispatch_table();

ref_counting_gc *rc_init(ptr start, size_t max_used);
void rc_free(ref_counting_gc *me);

bool rc_can_allot_p(ref_counting_gc *me, size_t n_bytes);
ptr rc_do_allot(ref_counting_gc *me, int type, size_t n_bytes);
size_t rc_space_used(ref_counting_gc *me);

#endif
