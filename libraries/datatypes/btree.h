// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef BTREE_H
#define BTREE_H

#include <stdbool.h>
#include "datatypes/common.h"

// The grand-daddy of trees, the motherload, the big whoop in the
// pond.

#define B_ORDER     8

typedef struct _bnode {
    struct _bnode *childs[B_ORDER + 1];
    int keys[B_ORDER];
    ptr vals[B_ORDER];
    size_t count;
} bnode;

typedef struct {
    bnode *root;
    size_t size;
} btree;

btree *btree_init();
void btree_free(btree *me);

bnode *btree_find(btree *me, int key, int *index);

bnode *bnode_init();
void bnode_free(bnode *me);

size_t bnode_linear_search(bnode *me, int key, bool *found);

#endif
