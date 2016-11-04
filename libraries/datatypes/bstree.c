#include <assert.h>
#include <stdio.h>
#include "bstree.h"

static bstree *
bst_init(bstree *parent, size_t key, ptr value) {
    bstree *me = (bstree *)malloc(sizeof(bstree));
    me->parent = parent;
    me->left = NULL;
    me->right = NULL;
    me->key = key;
    me->value = value;
    return me;
}

void
bst_free(bstree *bst) {
    if (bst) {
        bst_free(bst->left);
        bst_free(bst->right);
        free(bst);
    }
}

bstree *
bst_add(bstree *me, size_t key, ptr value) {
    bstree **addr = &me;
    bstree *parent = NULL;
    while (*addr) {
        ptr this_key = (*addr)->key;
        parent = *addr;
        if (key < this_key) {
            addr = &(*addr)->left;
        } else {
            addr = &(*addr)->right;
        }
    }
    *addr = bst_init(parent, key,value);
    return me;
}

// This algorithm is a little magic.
bstree *
bst_remove(bstree *root, bstree *z) {
    assert(root);
    assert(z);
    bstree *y;
    if (!z->left || !z->right) {
        // Node has only one child.
        y = z;
    } else {
        y = bst_min_node(z->right);
    }
    bstree *x;
    if (y->left) {
        x = y->left;
    } else {
        x = y->right;
    }
    if (x) {
        x->parent = y->parent;
    }
    if (!y->parent) {
        root = x;
    } else if (y == y->parent->left) {
        y->parent->left = x;
    } else {
        y->parent->right = x;
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
            me = me->left;
        } else if (key > me_key) {
            me = me->right;
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
            me = me->left;
        } else if (key > me_key) {
            me = me->right;
        } else {
            return me;
        }
    }
    return best;
}

bstree *
bst_min_node(bstree *me) {
    if (me) {
        while (me->left) {
            me = me->left;
        }
    }
    return me;
}

bstree *
bst_max_node(bstree *me) {
    if (me) {
        while (me->right) {
            me = me->right;
        }
    }
    return me;
}

bstree *
bst_successor(bstree *root, bstree *node) {
    if (!root) {
        return NULL;
    }
    if (!node) {
        return bst_min_node(root);
    }
    if (node->right) {
        return bst_min_node(node->right);
    }
    bstree *x = node->parent;
    while (x && node == x->right) {
        node = x;
        x = node->parent;
    }
    return x;
}

size_t
bst_size(bstree *bst) {
    if (!bst)
        return 0;
    return 1 + bst_size(bst->left) + bst_size(bst->right);
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
        bst_print(me->left, indent, print_null);
        bst_print(me->right, indent, print_null);
    }
}
