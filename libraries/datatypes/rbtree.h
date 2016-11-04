#ifndef RBTREE_H
#define RBTREE_H

#include <stdbool.h>
#include "datatypes/common.h"

typedef struct _rbtree {
    struct _rbtree *left, *right, *parent;
    ptr data;
    bool is_red;
} rbtree;

void rbt_free(rbtree *me);

// Tree mutation
rbtree *rbt_add(rbtree *me, ptr data);
rbtree *rbt_remove(rbtree *root, rbtree *node);

// Finding nodes
rbtree *rbt_find(rbtree *me, ptr data);
rbtree *rbt_find_lower_bound(rbtree *me, ptr data);
rbtree *rbt_successor(rbtree *root, rbtree *node);

// Tree stats
size_t rbt_black_height(rbtree *me);

// Dumping & Diagnostics
void rbt_print(rbtree *me, int indent, bool print_null);
void rbt_check_valid(rbtree *me);

#endif
