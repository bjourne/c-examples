// A QuickFit memory allocator based on Factor.
#include <limits.h>
#include "quickfit.h"

static void
qf_add_block(quick_fit *qf, ptr p, size_t size) {
    AT(p) = size;
    qf->n_blocks++;
    qf->free_space += size;
    int bucket = size / QF_DATA_ALIGNMENT;
    if (bucket < QF_N_BUCKETS) {
        v_add(qf->buckets[bucket], p);
    } else {
        v_add(qf->large_blocks, p);
    }
}

quick_fit *
qf_init(ptr start, size_t size) {
    quick_fit *qf = (quick_fit *)malloc(sizeof(quick_fit));
    qf->large_blocks = v_init(32);
    qf->start = start;
    qf->size = size;
    qf->n_blocks = 0;
    qf->free_space = 0;
    for (int i = 0; i < QF_N_BUCKETS; i++) {
        qf->buckets[i] = v_init(32);
    }
    qf_add_block(qf, start, size);
    return qf;
}

void
qf_free(quick_fit *qf) {
    for (int i = 0; i < QF_N_BUCKETS; i++) {
        v_free(qf->buckets[i]);
    }
    v_free(qf->large_blocks);
    free(qf);
}

static ptr
qf_find_large_block(quick_fit *qf, size_t req_size) {
    vector *v = qf->large_blocks;
    size_t best_size = UINT_MAX;
    int best_idx = 0;

    for (int i = 0; i < v->used; i++) {
        ptr el = v->array[i];
        size_t el_size = AT(el);
        if (el_size < best_size && el_size >= req_size) {
            best_size = el_size;
            best_idx = i;
        }
    }
    if (best_size < UINT_MAX) {
        qf->n_blocks--;
        qf->free_space -= best_size;
        return v_remove_at(v, best_idx);
    }
    return 0;
}

static void
qf_split_block(quick_fit *qf, ptr p, size_t req_size) {
    size_t block_size = AT(p);
    if (block_size > req_size) {
        ptr split = p + req_size;
        qf_add_block(qf, split, block_size - req_size);
        AT(p) = req_size;
    }
}

static ptr
qf_find_small_block(quick_fit *qf, size_t bucket, size_t req_size) {
    vector *v = qf->buckets[bucket];
    if (v->used == 0) {
        size_t large_size = QF_LARGE_BLOCK_SIZE(req_size);
        ptr p = qf_find_large_block(qf, large_size);
        if (!p)
            return 0;
        qf_split_block(qf, p, large_size);
        ptr end = p + large_size;
        for (; p < end; p += req_size) {
            qf_add_block(qf, p, req_size);
        }
    }
    qf->n_blocks--;
    qf->free_space -= req_size;
    ptr small_p = v_pop(v);
    return small_p;
}

static ptr
qf_find_free_block(quick_fit *qf, size_t size) {
    int bucket = size / QF_DATA_ALIGNMENT;
    if (bucket < QF_N_BUCKETS) {
        return qf_find_small_block(qf, bucket, size);
    }
    return qf_find_large_block(qf, size);
}

ptr
qf_allot_block(quick_fit *qf, size_t size) {
    size = ALIGN(size, QF_DATA_ALIGNMENT);
    ptr p = qf_find_free_block(qf, size);
    if (p) {
        qf_split_block(qf, p, size);
    }
    return p;
}

void
qf_free_block(quick_fit *qf, ptr p) {
    qf_add_block(qf, p, AT(p));
}

void
qf_print(quick_fit *qf) {
    ptr end = qf->start + qf->size;
    for (ptr p = qf->start; p < end; p += AT(p)) {
        printf("@%p: %4lu bytes\n", (void *)p, AT(p));
    }
}

size_t
qf_largest_free_block(quick_fit *qf) {
    vector *large_blocks = qf->large_blocks;
    if (large_blocks->used) {
        return AT(v_peek(large_blocks));
    }
    for (int i = QF_N_BUCKETS - 1; i >= 0; i--) {
        vector *small_blocks = qf->buckets[i];
        if (small_blocks->used) {
            return AT(v_peek(small_blocks));
        }
    }
    return 0;
}