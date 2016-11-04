#ifndef BSTREE_H
#define BSTREE_H

#include <stdbool.h>
#include "common.h"

typedef struct _bstree {
    struct _bstree *parent, *left, *right;
    size_t key;
    ptr value;
} bstree;

void bst_free(bstree *bst);

// Tree mutation
bstree *bst_add(bstree *me, size_t key, ptr value);

// The node must exist in the tree.
bstree *bst_remove(bstree *root, bstree *node);

// Finding nodes
bstree *bst_find(bstree *bst, size_t key);
bstree *bst_find_lower_bound(bstree *me, size_t key);

// If the tree is empty, NULL is returned.
bstree *bst_min_node(bstree *me);
bstree *bst_max_node(bstree *me);

bstree *bst_successor(bstree *root, bstree *node);

// Tree stats
size_t bst_size(bstree *bst);

// Dumping
void bst_print(bstree *me, int indent, bool print_null);

#endif
