// A QuickFit memory allocator based on Factor.
#include <limits.h>
#include "quickfit/quickfit.h"

void
qf_free_block(quick_fit *qf, ptr p, size_t size) {
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

void
qf_clear(quick_fit *qf) {
    qf->large_blocks->used = 0;
    qf->n_blocks = 0;
    qf->free_space = 0;
    for (int i = 0; i < QF_N_BUCKETS; i++) {
        qf->buckets[i]->used = 0;
    }
}

quick_fit *
qf_init(ptr start, size_t size) {
    quick_fit *qf = (quick_fit *)malloc(sizeof(quick_fit));
    qf->large_blocks = v_init(32);
    for (int i = 0; i < QF_N_BUCKETS; i++) {
        qf->buckets[i] = v_init(32);
    }
    qf_clear(qf);
    qf_free_block(qf, start, size);
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
        qf_free_block(qf, split, block_size - req_size);
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
            qf_free_block(qf, p, req_size);
        }
    }
    qf->n_blocks--;
    qf->free_space -= req_size;
    return v_remove(v);
}

static ptr
qf_find_free_block(quick_fit *me, size_t size) {
    int bucket = size / QF_DATA_ALIGNMENT;
    if (bucket < QF_N_BUCKETS) {
        return qf_find_small_block(me, bucket, size);
    }
    return qf_find_large_block(me, size);
}

ptr
qf_allot_block(quick_fit *me, size_t size) {
    size = ALIGN(size, QF_DATA_ALIGNMENT);
    ptr p = qf_find_free_block(me, size);
    if (p) {
        qf_split_block(me, p, size);
    }
    return p;
}

void
qf_print(quick_fit *qf, ptr start, size_t size) {
    ptr end = start + size;
    for (ptr iter = start; iter < end; iter += AT(iter)) {
        printf("@%5lu: %4lu\n", iter - start, AT(iter));
    }
}

bool
qf_can_allot_p(quick_fit *me, size_t size) {
    size_t small = ALIGN(size, QF_DATA_ALIGNMENT);
    int bucket = small / QF_DATA_ALIGNMENT;
    if (bucket < QF_N_BUCKETS) {
        if (me->buckets[bucket]->used > 0) {
            return true;
        }
        size = QF_LARGE_BLOCK_SIZE(small);
    }
    for (int i = 0; i < me->large_blocks->used; i++) {
        ptr p = me->large_blocks->array[i];
        if (AT(p) >= size)
            return true;
    }
    return false;
}