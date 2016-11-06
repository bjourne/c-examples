#include <string.h>
#include "datatypes/btree.h"

bnode *
bnode_init() {
    bnode *me = (bnode *)malloc(sizeof(bnode));
    me->count = 0;
    memset(me->childs, 0, sizeof(me->childs));
    return me;
}

void
bnode_free(bnode *me) {
    if (me) {
        for (int i = 0; i < me->count; i++) {
            bnode_free(me->childs[i]);
        }
        free(me);
    }
}

int
bnode_linear_search(bnode *me, int key, bool *found) {
    *found = false;
    int i;
    for (i = 0; i < me->count; i++) {
        int this_key = me->keys[i];
        if (this_key > key) {
            return i;
        } else if (this_key == key) {
            *found = true;
            return i;
        }
    }
    return i;
}

bnode *
btree_find(btree *me, int key, int *index) {
    bool found;


    bnode *iter = me->root;
    while (iter) {
        *index = bnode_linear_search(iter, key, &found);
        if (found) {
            return iter;
        }
        iter = iter->childs[*index];
    }
    return iter;
}

btree *
btree_init() {
    btree *me = (btree *)malloc(sizeof(btree));
    me->root = NULL;
    return me;
}

void
btree_free(btree *me) {
    bnode_free(me->root);
    free(me);
}
