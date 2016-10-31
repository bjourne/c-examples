#ifndef RBTREE_H
#define RBTREE_H

#include <stdbool.h>
#include "datatypes/common.h"

typedef struct _rbtree {
    struct _rbtree *left, *right, *parent;
    ptr data;
    bool is_red;
} rbtree;

rbtree *rbt_add(rbtree *rbt, ptr data);
void rbt_free(rbtree *me);

void rbt_print(rbtree *me, int indent, bool print_null);

#endif
