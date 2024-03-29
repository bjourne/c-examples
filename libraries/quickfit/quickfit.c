// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// A QuickFit memory allocator based on Factor.
#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include "quickfit/quickfit.h"

void
qf_free_block(quick_fit *qf, ptr p, size_t size) {
    QF_SET_BLOCK_SIZE(p, size);
    qf->n_blocks++;
    qf->free_space += size;
    size_t bucket = size / QF_DATA_ALIGNMENT;
    if (bucket < QF_N_BUCKETS) {
        v_add(qf->buckets[bucket], p);
    } else {
        qf->large_blocks = rbt_add(qf->large_blocks, (bstkey)size, p);
    }
}

void
qf_clear(quick_fit *qf) {
    rbt_free(qf->large_blocks);
    qf->large_blocks = NULL;
    qf->n_blocks = 0;
    qf->free_space = 0;
    for (int i = 0; i < QF_N_BUCKETS; i++) {
        qf->buckets[i]->used = 0;
    }
}

quick_fit *
qf_init(ptr start, size_t size) {
    quick_fit *qf = (quick_fit *)malloc(sizeof(quick_fit));
    qf->start = start;
    qf->size = size;
    qf->large_blocks = NULL;
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
    rbt_free(qf->large_blocks);
    free(qf);
}

static ptr
qf_find_large_block(quick_fit *qf, size_t req_size) {
    rbtree *node = rbt_find_lower_bound(qf->large_blocks, (bstkey)req_size);
    if (node) {
        qf->n_blocks--;
        qf->free_space -= node->key;
        ptr p = node->value;
        qf->large_blocks = rbt_remove(qf->large_blocks, node);
        return p;
    }
    return 0;
}

static void
qf_split_block(quick_fit *qf, ptr p, size_t req_size) {
    size_t block_size = QF_GET_BLOCK_SIZE(p);
    if (block_size > req_size) {
        ptr split = p + req_size;
        qf_free_block(qf, split, block_size - req_size);
        QF_SET_BLOCK_SIZE(p, req_size);
    }
}

static ptr
qf_find_small_block(quick_fit *qf, size_t bucket, size_t req_size) {
    assert(req_size == bucket * QF_DATA_ALIGNMENT);
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
    size_t bucket = size / QF_DATA_ALIGNMENT;
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
qf_print(quick_fit *me) {
    ptr end = me->start + me->size;
    ptr iter = me->start;
    while (iter < end) {
        size_t size = QF_GET_BLOCK_SIZE(iter);
        ptr rel_addr = iter - me->start;
        printf("@%5" PRIuPTR ": %4llu\n",
               rel_addr, (unsigned long long)size);
        iter += size;
    }
}

size_t
qf_space_used(quick_fit *me) {
    return me->size - me->free_space;
}

bool
qf_can_allot_p(quick_fit *me, size_t size) {
    size_t small = ALIGN(size, QF_DATA_ALIGNMENT);
    size_t bucket = small / QF_DATA_ALIGNMENT;
    if (bucket < QF_N_BUCKETS) {
        if (me->buckets[bucket]->used > 0) {
            return true;
        }
        size = QF_LARGE_BLOCK_SIZE(small);
    }
    rbtree *node = rbt_iterate(me->large_blocks, NULL, BST_RIGHT);
    if (size > BST_KEY_MAX) {
        return false;
    }
    if (node && node->key >= (bstkey)size) {
        return true;
    }
    return false;
}
