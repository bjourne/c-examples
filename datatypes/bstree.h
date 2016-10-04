#ifndef BSTREE_H
#define BSTREE_H

#include "common.h"

typedef struct _bstree {
    ptr data;
    struct _bstree *left, *right;
} bstree;

void bst_free(bstree *bst);

bstree *bst_add(bstree *bst, ptr data);
bstree *bst_remove(bstree *bst, ptr data);

bstree *bst_find(bstree *bst, ptr data);
bstree *bst_min_node(bstree *bst);
bstree *bst_max_node(bstree *bst);
size_t bst_size(bstree *bst);

void bst_print_inorder(bstree *bst);

#endif
