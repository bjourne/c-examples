#ifndef QUICKFIT_H
#define QUICKFIT_H

#include <stdbool.h>
#include "datatypes/bits.h"
#include "datatypes/rbtree.h"
#include "datatypes/vector.h"

#define QF_N_BUCKETS 32
#define QF_DATA_ALIGNMENT 16
#define QF_PAGE_SIZE 1024
// small_size should be aligned.
#define QF_LARGE_BLOCK_SIZE(small_size) \
    ((QF_PAGE_SIZE + small_size - 1) / small_size) * small_size

// The header format is setup so that it is compatible with the one
// described in collectors/common.h.
#define QF_GET_BLOCK_SIZE(p)        BF_GET(AT(p), 8, 24)
#define QF_SET_BLOCK_SIZE(p, n)     AT(p) = BF_SET(AT(p), n, 8, 24)

typedef struct {
    vector* buckets[QF_N_BUCKETS];
    rbtree* large_blocks;
    size_t n_blocks;
    size_t free_space;
} quick_fit;

quick_fit *qf_init(ptr start, size_t size);
void qf_free(quick_fit *qf);

void qf_clear(quick_fit *qf);
ptr qf_allot_block(quick_fit *qf, size_t size);
void qf_free_block(quick_fit *qf, ptr p, size_t size);
void qf_print(quick_fit *qf, ptr start, size_t size);
bool qf_can_allot_p(quick_fit *qf, size_t size);

#endif
