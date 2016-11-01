#include <stdio.h>
#include "bstree.h"

static bstree *
bst_init(ptr data) {
    bstree *bst = (bstree *)malloc(sizeof(bstree));
    bst->left = NULL;
    bst->right = NULL;
    bst->data = data;
    return bst;
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
    while (*addr) {
        ptr this_data = (*addr)->data;
        if (data < this_data) {
            addr = &(*addr)->left;
        } else {
            addr = &(*addr)->right;
        }
    }
    *addr = bst_init(data);
    return me;
}

bstree *
bst_remove(bstree *me, ptr data) {
    if (!me) {
        return me;
    }
    if (data < me->data) {
        me->left = bst_remove(me->left, data);
    } else if (data > me->data) {
        me->right = bst_remove(me->right, data);
    } else {
        // Found node to delete
        if (!me->left) {
            bstree *right = me->right;
            me->right = NULL;
            free(me);
            me = right;
        } else if (!me->right) {
            bstree *left = me->left;
            me->left = NULL;
            free(me);
            me = left;
        } else {
            // It has two children. Copy inorder successors value.
            bstree *inorder_succ = bst_min_node(me->right);
            me->data = inorder_succ->data;
            me->right = bst_remove(me->right, inorder_succ->data);
        }
    }
    return me;
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
bst_min_node(bstree *bst) {
    while (bst->left) {
        bst = bst->left;
    }
    return bst;
}

bstree *
bst_max_node(bstree *bst) {
    while (bst->right) {
        bst = bst->right;
    }
    return bst;
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
