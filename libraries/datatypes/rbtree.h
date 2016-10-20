#ifndef RBTREE_H
#define RBTREE_H

typedef struct _rbtree {
    bool is_red;
    ptr data;
    struct _bstree *left, *right;
} bstree;

#endif
