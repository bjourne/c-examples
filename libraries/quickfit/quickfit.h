#ifndef QUICKFIT_H
#define QUICKFIT_H

#include <stdbool.h>
#include "datatypes/vector.h"

#define QF_N_BUCKETS 32
#define QF_DATA_ALIGNMENT 16
#define QF_PAGE_SIZE 1024
// small_size should be aligned.
#define QF_LARGE_BLOCK_SIZE(small_size) \
    ((QF_PAGE_SIZE + small_size - 1) / small_size) * small_size

typedef struct {
    vector* buckets[QF_N_BUCKETS];
    // Need a better datastructure here
    vector* large_blocks;
    size_t n_blocks;
    size_t free_space;
} quick_fit;

quick_fit *qf_init(ptr start, size_t size);
void qf_free(quick_fit *qf);

void qf_clear(quick_fit *qf);
ptr qf_allot_block(quick_fit *qf, size_t size);
void qf_free_block(quick_fit *qf, ptr p);
void qf_print(quick_fit *qf, ptr start, size_t size);
size_t qf_largest_free_block(quick_fit *qf);
bool qf_can_allot_p(quick_fit *qf, size_t size);

#endif
