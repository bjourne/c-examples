#ifndef RBTREE_H
#define RBTREE_H

#include <stdbool.h>
#include "datatypes/common.h"

typedef enum {
    RB_LEFT = 0,
    RB_RIGHT = 1
} rbdir;

typedef struct _rbtree {
    struct _rbtree *parent;
    struct _rbtree *childs[2];
    bool is_red;
    size_t key;
    ptr value;
} rbtree;

void rbt_free(rbtree *me);

// Tree mutation
rbtree *rbt_add(rbtree *me, size_t key, ptr value);
rbtree *rbt_remove(rbtree *root, rbtree *node);

// Finding nodes
rbtree *rbt_find(rbtree *me, size_t key);
rbtree *rbt_find_lower_bound(rbtree *me, size_t key);
rbtree *rbt_iterate(rbtree *root, rbtree *node, rbdir dir);

// Tree stats
size_t rbt_size(rbtree *bst);
size_t rbt_black_height(rbtree *me);

// Dumping & Diagnostics
void rbt_print(rbtree *me, int indent, bool print_null);
void rbt_check_valid(rbtree *me);

#endif
