#include <assert.h>
#include "datatypes/rbtree.h"

static rbtree *
rbt_init(rbtree *parent, ptr data) {
    rbtree *me = (rbtree *)malloc(sizeof(rbtree));
    me->left = NULL;
    me->right = NULL;
    me->parent = parent;
    me->data = data;
    me->is_red = true;
    return me;
}

void
rbt_free(rbtree *me) {
    if (me) {
        rbt_free(me->left);
        rbt_free(me->right);
        free(me);
    }
}

rbtree *
rbt_rotate_left(rbtree *root, rbtree *node) {
    rbtree *right_child = node->right;

    // Turn right_child's left sub-tree into node's right sub-tree */
    node->right = right_child->left;
    if (right_child->left) {
        right_child->left->parent = node;
    }

    // right_child's new parent was node's parent */
    right_child->parent = node->parent;
    if (!node->parent) {
        root = right_child;
    } else {
        if (node == (node->parent)->left) {
            node->parent->left = right_child;
        } else {
            node->parent->right = right_child;
        }
    }
    right_child->left = node;
    node->parent = right_child;
    return root;
}

rbtree *
rbt_rotate_right(rbtree *root, rbtree *node) {
    rbtree *left_child = node->left;

    // Turn left_child's right sub-tree into node's left sub-tree */
    node->left = left_child->right;
    if (left_child->right) {
        left_child->right->parent = node;
    }

    // left_child's new parent was node's parent */
    left_child->parent = node->parent;
    if (!node->parent) {
        root = left_child;
    } else {
        if (node == (node->parent)->right) {
            node->parent->right = left_child;
        } else {
            node->parent->left = left_child;
        }
    }
    left_child->right = node;
    node->parent = left_child;
    return root;
}


static rbtree *
rbt_uncle(rbtree *node) {
    rbtree *p = node->parent;
    rbtree *gp = p->parent;
    return p == gp->left ? gp->right : gp->left;
}

static rbtree *
rbt_fixup(rbtree *root, rbtree *node) {
    if (node != root && node->parent->is_red) {
        rbtree *uncle = rbt_uncle(node);
        if (uncle && uncle->is_red) {
            node->parent->is_red = false;
            uncle->is_red = false;
            node->parent->parent->is_red = true;
            return rbt_fixup(root, node->parent->parent);
        } else if (node->parent == node->parent->parent->left) {
            if (node == node->parent->right) {
                node = node->parent;
                root = rbt_rotate_left(root, node);
            }
            node->parent->is_red = false;
            node->parent->parent->is_red = true;
            root = rbt_rotate_right(root, node->parent->parent);
        } else if (node->parent == node->parent->parent->right) {
            if (node == node->parent->left) {
                node = node->parent;
                root = rbt_rotate_right(root, node);
            }
            node->parent->is_red = false;
            node->parent->parent->is_red = true;
            root = rbt_rotate_left(root, node->parent->parent);
        }
    }
    root->is_red = false;
    return root;
}

rbtree *
rbt_add(rbtree *me, ptr data) {
    // Find insertion point.
    rbtree **addr = &me;
    rbtree *parent = NULL;
    while (*addr) {
        ptr this_data = (*addr)->data;
        parent = *addr;
        if (data < this_data) {
            addr = &(*addr)->left;
        } else {
            addr = &(*addr)->right;
        }
    }
    *addr = rbt_init(parent, data);
    return rbt_fixup(me, *addr);
}

void
rbt_print(rbtree *me, int indent, bool print_null) {
    if (!me) {
        if (print_null) {
            printf("%*sNULL\n", indent, "");
        }
    } else {
        printf("%*s%lu %s\n", indent, "", me->data, me->is_red ? "R" : "B");
        indent += 2;
        rbt_print(me->left, indent, print_null);
        rbt_print(me->right, indent, print_null);
    }
}
