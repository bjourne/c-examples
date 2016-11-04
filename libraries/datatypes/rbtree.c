#include <assert.h>
#include "datatypes/rbtree.h"

static rbtree *
rbt_init(rbtree *parent, ptr data) {
    rbtree *me = (rbtree *)malloc(sizeof(rbtree));
    me->parent = parent;
    me->left = NULL;
    me->right = NULL;
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
rbt_add_fixup(rbtree *root, rbtree *node) {
    if (node != root && node->parent->is_red) {
        rbtree *uncle = rbt_uncle(node);
        if (uncle && uncle->is_red) {
            node->parent->is_red = false;
            uncle->is_red = false;
            node->parent->parent->is_red = true;
            return rbt_add_fixup(root, node->parent->parent);
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
    return rbt_add_fixup(me, *addr);
}

rbtree *
rbt_find(rbtree *me, ptr data) {
    if (!me) {
        return NULL;
    }
    ptr me_data = me->data;
    if (data < me_data) {
        return rbt_find(me->left, data);
    } else if (data > me_data) {
        return rbt_find(me->right, data);
    }
    return me;
}

rbtree *
rbt_min_node(rbtree *me) {
    while (me->left) {
        me = me->left;
    }
    return me;
}

rbtree *
rbt_max_node(rbtree *me) {
    while (me->right) {
        me = me->right;
    }
    return me;
}

#define IS_BLACK(n)     (!(n) || !(n)->is_red)
#define BOTH_CHILDREN_BLACK(n)   (!(n) || (IS_BLACK((n)->left) && IS_BLACK((n)->right)))

// I can't claim to understand this algorithm. It is mostly
// transliterated from
// https://github.com/headius/redblack/blob/master/red_black_tree.py
// and https://github.com/codekenq/Red-Black-Tree.git.
//
// I should have used sentinel nodes for NULL. It would make the
// algorithm much easier to read.
static rbtree *
rbt_remove_fixup(rbtree *root, rbtree *x, rbtree *x_parent) {
    while (x != root && (!x || !x->is_red)) {
        // w is x's sibling */
        rbtree *w;
        if (x == x_parent->left) {
            w = x_parent->right;
            if (w && w->is_red) {
                w->is_red = false;
                x_parent->is_red = true;
                root = rbt_rotate_left(root, x_parent);
                w = x_parent->right;
            }
            if (BOTH_CHILDREN_BLACK(w)) {
                if (w) {
                    w->is_red = true;
                }
                x = x_parent;
                x_parent = x->parent;
            } else {
                if (IS_BLACK(w->right)) {
                    w->left->is_red = false;
                    w->is_red = true;
                    root = rbt_rotate_right(root, w);
                    w = x_parent->right;
                }
                w->is_red = x_parent->is_red;
                x_parent->is_red = false;
                w->right->is_red = false;
                root = rbt_rotate_left(root, x_parent);
                x = root;
            }
        } else {
            w = x_parent->left;
            if (w && w->is_red) {
                w->is_red = false;
                x_parent->is_red = true;
                root = rbt_rotate_right(root, x_parent);
                w = x_parent->left;
            }
            if (BOTH_CHILDREN_BLACK(w)) {
                if (w) {
                    w->is_red = true;
                }
                x = x_parent;
                x_parent = x->parent;
            } else {
                if (IS_BLACK(w->left)) {
                    w->right->is_red = false;
                    w->is_red = true;
                    root = rbt_rotate_left(root, w);
                    w = x_parent->left;
                }
                w->is_red = x_parent->is_red;
                x_parent->is_red = false;
                w->left->is_red = false;
                root = rbt_rotate_right(root, x_parent);
                x = root;
            }
        }
    }
    if (x) {
        x->is_red = false;
    }
    return root;
}

rbtree *
rbt_remove(rbtree *root, rbtree *z) {
    assert(z);
    // y is the successor sometimes.
    rbtree *y;
    if (!z->left || !z->right) {
        y = z;
    } else {
        // It has two children. Copy inorder successors value.
        y = rbt_min_node(z->right);
    }
    rbtree *x;
    if (!y->left) {
        x = y->right;
    } else {
        x = y->left;
    }
    if (x) {
        x->parent = y->parent;
    }
    if (!y->parent) {
        root = x;
    } else {
        if (y == y->parent->left) {
            y->parent->left = x;
        } else {
            y->parent->right = x;
        }
    }
    if (y != z) {
        z->data = y->data;
    }
    if (!y->is_red) {
        root = rbt_remove_fixup(root, x, y->parent);

    }
    free(y);
    return root;
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

size_t
rbt_black_height(rbtree *me) {
    if (!me) {
        return 1;
    }
    size_t left_height = rbt_black_height(me->left);
    return (me->is_red ? 0 : 1) + left_height;
}

rbtree *
rbt_successor(rbtree *root, rbtree *node) {
    if (!root) {
        return NULL;
    }
    if (!node) {
        return rbt_min_node(root);
    }
    if (node->right) {
        return rbt_min_node(node->right);
    }
    rbtree *x = node->parent;
    while (x && node == x->right) {
        node = x;
        x = node->parent;
    }
    return x;
}

void
rbt_check_valid(rbtree *me) {
    if (!me) {
        return;
    }
    size_t left_height = 1;
    rbtree *left = me->left;
    rbtree *right = me->right;
    bool is_red = me->is_red;
    if (left) {
        assert((is_red && !left->is_red) || !is_red);
        assert(left->data <= me->data);
        rbt_check_valid(left);
        left_height = rbt_black_height(left);
    }
    size_t right_height = 1;
    if (me->right) {
        assert((is_red && !right->is_red) || !is_red);
        assert(right->data >= me->data);
        rbt_check_valid(right);
        right_height = rbt_black_height(right);
    }
    assert(left_height == right_height);
}
