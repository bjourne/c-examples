#ifndef COLLECTORS_COPYING2_H
#define COLLECTORS_COPYING2_H

void cg_collect_optimized(copying_gc *me, vector* roots);

// Interface support
gc_dispatch *cg_get_dispatch_table_optimized();

#endif
