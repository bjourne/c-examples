#ifndef BSTREE_H
#define BSTREE_H

#include <stdbool.h>
#include "common.h"

typedef struct _bstree {
    struct _bstree *left, *right;
    ptr data;
} bstree;

void bst_free(bstree *bst);

bstree *bst_add(bstree *bst, ptr data);
bstree *bst_remove(bstree *bst, ptr data);

bstree *bst_find(bstree *bst, ptr data);
bstree *bst_min_node(bstree *bst);
bstree *bst_max_node(bstree *bst);
size_t bst_size(bstree *bst);

void bst_print(bstree *me, int indent, bool print_null);

#endif
