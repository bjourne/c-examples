#include <assert.h>
#include <stdio.h>
#include "bstree.h"

static bstree *
bst_init(bstree *parent, ptr data) {
    bstree *me = (bstree *)malloc(sizeof(bstree));
    me->parent = parent;
    me->left = NULL;
    me->right = NULL;
    me->data = data;
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
bst_add(bstree *me, ptr data) {
    bstree **addr = &me;
    bstree *parent = NULL;
    while (*addr) {
        ptr this_data = (*addr)->data;
        parent = *addr;
        if (data < this_data) {
            addr = &(*addr)->left;
        } else {
            addr = &(*addr)->right;
        }
    }
    *addr = bst_init(parent, data);
    return me;
}

// This algorithm is a litte magic.
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
    if (y != z) {
        ptr y_data = y->data;
        y->data = z->data;
        z->data = y_data;
    }
    free(y);
    return root;
}

bstree *
bst_find(bstree *me, ptr data) {
    if (!me) {
        return NULL;
    }
    ptr me_data = me->data;
    if (data < me_data) {
        return bst_find(me->left, data);
    } else if (data > me_data) {
        return bst_find(me->right, data);
    }
    return me;
}

bstree *
bst_find_lower_bound(bstree *me, ptr data, bstree *best) {
    if (!me) {
        return best;
    }
    ptr me_data = me->data;
    if (me_data >= data && (!best || me_data < best->data)) {
        best = me;
    }
    if (data < me_data) {
        return bst_find_lower_bound(me->left, data, best);
    } else if (data > me_data) {
        return bst_find_lower_bound(me->right, data, best);
    }
    return best;
}

bstree *
bst_min_node(bstree *me) {
    assert(me);
    while (me->left) {
        me = me->left;
    }
    return me;
}

bstree *
bst_max_node(bstree *me) {
    assert(me);
    while (me->right) {
        me = me->right;
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
        printf("%*s%lu\n", indent, "", me->data);
        indent += 2;
        bst_print(me->left, indent, print_null);
        bst_print(me->right, indent, print_null);
    }
}
