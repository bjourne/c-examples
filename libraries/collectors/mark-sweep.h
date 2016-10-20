#ifndef MARK_SWEEP_H
#define MARK_SWEEP_H

#define MS_GET_MARK(p) (AT(p) & 1)

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

#endif
