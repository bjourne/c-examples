#include <assert.h>
#include <stdio.h>
#include "bstree.h"

static bstree *
bst_init(bstree *parent, size_t key, ptr value) {
    bstree *me = (bstree *)malloc(sizeof(bstree));
    me->parent = parent;
    me->childs[BST_LEFT] = NULL;
    me->childs[BST_RIGHT] = NULL;
    me->key = key;
    me->value = value;
    return me;
}

void
bst_free(bstree *bst) {
    if (bst) {
        bst_free(bst->childs[BST_LEFT]);
        bst_free(bst->childs[BST_RIGHT]);
        free(bst);
    }
}

bstree *
bst_add(bstree *me, size_t key, ptr value) {
    bstree **addr = &me;
    bstree *parent = NULL;
    while (*addr) {
        parent = *addr;
        addr = &parent->childs[key >= parent->key];
    }
    *addr = bst_init(parent, key, value);
    return me;
}

static bstree *
bst_extreme_node(bstree *me, bstdir dir) {
    while (me->childs[dir]) {
        me = me->childs[dir];
    }
    return me;
}

// This algorithm is a little magic.
bstree *
bst_remove(bstree *root, bstree *z) {
    assert(root);
    assert(z);
    bstree *y;
    if (!z->childs[BST_LEFT] || !z->childs[BST_RIGHT]) {
        // Node has only one child.
        y = z;
    } else {
        y = bst_extreme_node(z->childs[BST_RIGHT], BST_LEFT);
    }
    bstree *x = y->childs[BST_RIGHT];
    if (!x) {
        x = y->childs[BST_LEFT];
    }
    if (x) {
        x->parent = y->parent;
    }
    if (!y->parent) {
        root = x;
    } else {
        y->parent->childs[BST_DIR_OF(y)] = x;
    }
    z->key = y->key;
    z->value = y->value;
    free(y);
    return root;
}

bstree *
bst_find(bstree *me, size_t key) {
    while (me) {
        ptr me_key = me->key;
        if (key < me_key) {
            me = me->childs[BST_LEFT];
        } else if (key > me_key) {
            me = me->childs[BST_RIGHT];
        } else {
            return me;
        }
    }
    return me;
}

bstree *
bst_find_lower_bound(bstree *me, size_t key) {
    bstree *best = NULL;
    size_t best_key = UINTPTR_MAX;
    while (me) {
        ptr me_key = me->key;
        if (key < me_key) {
            if (me_key < best_key) {
                best_key = me_key;
                best = me;
            }
            me = me->childs[BST_LEFT];
        } else if (key > me_key) {
            me = me->childs[BST_RIGHT];
        } else {
            return me;
        }
    }
    return best;
}

bstree *
bst_iterate(bstree *root, bstree *node, bstdir dir) {
    if (!root) {
        return NULL;
    }
    if (!node) {
        return bst_extreme_node(root, dir);
    }
    if (node->childs[!dir]) {
        return bst_extreme_node(node->childs[!dir], dir);
    }
    bstree *x = node->parent;
    while (x && node == x->childs[!dir]) {
        node = x;
        x = node->parent;
    }
    return x;
}

size_t
bst_size(bstree *bst) {
    if (!bst)
        return 0;
    return 1 + bst_size(bst->childs[BST_LEFT]) + bst_size(bst->childs[BST_RIGHT]);
}

void
bst_print(bstree *me, int indent, bool print_null) {
    if (!me) {
        if (print_null) {
            printf("%*sNULL\n", indent, "");
        }
    } else {
        printf("%*s%lu\n", indent, "", me->key);
        indent += 2;
        bst_print(me->childs[BST_LEFT], indent, print_null);
        bst_print(me->childs[BST_RIGHT], indent, print_null);
    }
}
