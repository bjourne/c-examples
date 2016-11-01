#ifndef BSTREE_H
#define BSTREE_H

#include <stdbool.h>
#include "common.h"

typedef struct _bstree {
    struct _bstree *left, *right, *parent;
    ptr data;
} bstree;

void bst_free(bstree *bst);

// Tree mutation
bstree *bst_add(bstree *me, ptr data);

// The node must exist in the tree.
bstree *bst_remove(bstree *root, bstree *node);

// Finding nodes
bstree *bst_find(bstree *bst, ptr data);
bstree *bst_find_lower_bound(bstree *me, ptr data, bstree *best);

// Tree must not be empty.
bstree *bst_min_node(bstree *me);
bstree *bst_max_node(bstree *me);

bstree *bst_successor(bstree *root, bstree *node);

// Info
size_t bst_size(bstree *bst);

// Dumping
void bst_print(bstree *me, int indent, bool print_null);

#endif
