#ifndef BSTREE_H
#define BSTREE_H

#include <stdbool.h>
#include "datatypes/common.h"
#include "datatypes/trees.h"

typedef struct _bstree {
    struct _bstree *parent;
    struct _bstree *childs[2];
    bstkey key;
    ptr value;
} bstree;

void bst_free(bstree *bst);

// Tree mutation
bstree *bst_add(bstree *me, bstkey key, ptr value);

// The node must exist in the tree.
bstree *bst_remove(bstree *root, bstree *node);

// Finding nodes
bstree *bst_find(bstree *bst, bstkey key);
bstree *bst_find_lower_bound(bstree *me, bstkey key);
bstree *bst_iterate(bstree *me, bstree *node, bstdir dir);

// Tree stats
size_t bst_size(bstree *bst);

// Dumping
void bst_print(bstree *me, int indent, bool print_null);

#endif
