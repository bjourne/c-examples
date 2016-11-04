#include <assert.h>
#include "datatypes/rbtree.h"

static rbtree *
rbt_init(rbtree *parent, size_t key, ptr value) {
    rbtree *me = (rbtree *)malloc(sizeof(rbtree));
    me->parent = parent;
    me->childs[RB_LEFT] = NULL;
    me->childs[RB_RIGHT] = NULL;
    me->is_red = true;
    me->key = key;
    me->value = value;
    return me;
}

void
rbt_free(rbtree *me) {
    if (me) {
        rbt_free(me->childs[RB_LEFT]);
        rbt_free(me->childs[RB_RIGHT]);
        free(me);
    }
}

// Determines the direction of n in relation to p. n is allowed to be NULL.
#define DIR_OF2(n, p)   ((n) == (p)->childs[RB_LEFT] ? RB_LEFT : RB_RIGHT)

// Determines if n is a left or right child. n can't be root because
// that node isn't a child.
#define DIR_OF(n)       DIR_OF2(n, (n->parent))

static rbtree *
rbt_rotate(rbtree *root, rbtree *node, rbdir dir) {
    rbtree *opp_child = node->childs[!dir];

    // Turn right_child's left sub-tree into node's right sub-tree */
    node->childs[!dir] = opp_child->childs[dir];
    if (opp_child->childs[dir]) {
        opp_child->childs[dir]->parent = node;
    }

    // opp_child's new parent was node's parent */
    opp_child->parent = node->parent;
    if (!node->parent) {
        root = opp_child;
    } else {
        node->parent->childs[DIR_OF(node)] = opp_child;
    }
    opp_child->childs[dir] = node;
    node->parent = opp_child;
    return root;
}

static rbtree *
rbt_add_fixup(rbtree *root, rbtree *x) {
    while (x != root && x->parent->is_red) {
        // dir is the direction of x's parent in relation to x's
        // grandparent.
        rbdir dir = DIR_OF(x->parent);
        rbtree *y = x->parent->parent->childs[!dir];
        if (y && y->is_red) {
            // Simple recoloring case.
            x->parent->is_red = false;
            y->is_red = false;
            x->parent->parent->is_red = true;
            x = x->parent->parent;
        } else {
            if (DIR_OF(x) != dir) {
                x = x->parent;
                root = rbt_rotate(root, x, dir);
            }
            x->parent->is_red = false;
            x->parent->parent->is_red = true;
            root = rbt_rotate(root, x->parent->parent, !dir);
        }
    }
    root->is_red = false;
    return root;
}

rbtree *
rbt_add(rbtree *me, size_t key, ptr value) {
    // Find insertion point.
    rbtree **addr = &me;
    rbtree *parent = NULL;
    while (*addr) {
        ptr this_key = (*addr)->key;
        parent = *addr;
        if (key < this_key) {
            addr = &(*addr)->childs[RB_LEFT];
        } else {
            addr = &(*addr)->childs[RB_RIGHT];
        }
    }
    *addr = rbt_init(parent, key, value);
    return rbt_add_fixup(me, *addr);
}

rbtree *
rbt_find(rbtree *me, size_t key) {
    while (me) {
        ptr me_key = me->key;
        if (key < me_key) {
            me = me->childs[RB_LEFT];
        } else if (key > me_key) {
            me = me->childs[RB_RIGHT];
        } else {
            return me;
        }
    }
    return me;
}

static rbtree *
rbt_extreme_node(rbtree *me, rbdir dir) {
    while (me->childs[dir]) {
        me = me->childs[dir];
    }
    return me;
}

#define IS_BLACK(n)     (!(n) || !(n)->is_red)
#define BOTH_CHILDREN_BLACK(n)   (!(n) || (IS_BLACK((n)->childs[RB_LEFT]) && IS_BLACK((n)->childs[RB_RIGHT])))

// I can't claim to understand this algorithm. It is mostly
// transliterated from
// https://github.com/headius/redblack/blob/master/red_black_tree.py
// and https://github.com/codekenq/Red-Black-Tree.git.
//
// I should have used sentinel nodes for NULL. It would make the
// algorithm much easier to read.
//
// Note that x can be NULL and then of course x_parent != x->parent.
static rbtree *
rbt_remove_fixup(rbtree *root, rbtree *x, rbtree *x_parent) {
    while (x != root && IS_BLACK(x)) {
        rbdir dir = DIR_OF2(x, x_parent);
        rbtree *w = x_parent->childs[!dir];
        if (w && w->is_red) {
            w->is_red = false;
            x_parent->is_red = true;
            root = rbt_rotate(root, x_parent, dir);
            w = x_parent->childs[!dir];
        }
        if (BOTH_CHILDREN_BLACK(w)) {
            if (w) {
                w->is_red = true;
            }
            x = x_parent;
            x_parent = x->parent;
        } else {
            if (IS_BLACK(w->childs[!dir])) {
                w->childs[dir]->is_red = false;
                w->is_red = true;
                root = rbt_rotate(root, w, !dir);
                w = x_parent->childs[!dir];
            }
            w->is_red = x_parent->is_red;
            x_parent->is_red = false;
            w->childs[!dir]->is_red = false;
            root = rbt_rotate(root, x_parent, dir);
            x = root;
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
    if (!z->childs[RB_LEFT] || !z->childs[RB_RIGHT]) {
        y = z;
    } else {
        // It has two children. Copy inorder successors value.
        y = rbt_extreme_node(z->childs[RB_RIGHT], RB_LEFT);
    }
    rbtree *x;
    if (!y->childs[RB_LEFT]) {
        x = y->childs[RB_RIGHT];
    } else {
        x = y->childs[RB_LEFT];
    }
    if (x) {
        x->parent = y->parent;
    }
    if (!y->parent) {
        root = x;
    } else {
        y->parent->childs[DIR_OF(y)] = x;
    }
    if (y != z) {
        z->key = y->key;
        z->value = y->value;
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
        printf("%*s%lu %s\n", indent, "", me->key, me->is_red ? "R" : "B");
        indent += 2;
        rbt_print(me->childs[RB_LEFT], indent, print_null);
        rbt_print(me->childs[RB_RIGHT], indent, print_null);
    }
}

size_t
rbt_black_height(rbtree *me) {
    if (!me) {
        return 1;
    }
    size_t left_height = rbt_black_height(me->childs[RB_LEFT]);
    return (me->is_red ? 0 : 1) + left_height;
}

rbtree *
rbt_iterate(rbtree *root, rbtree *node, rbdir dir) {
    if (!root) {
        return NULL;
    }
    if (!node) {
        return rbt_extreme_node(root, dir);
    }
    if (node->childs[!dir]) {
        return rbt_extreme_node(node->childs[!dir], dir);
    }
    rbtree *x = node->parent;
    while (x && node == x->childs[!dir]) {
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
    rbtree *left = me->childs[RB_LEFT];
    rbtree *right = me->childs[RB_RIGHT];
    if (me->is_red) {
        assert(!left || !left->is_red);
        assert(!right || !right->is_red);
    }
    if (left) {
        assert(left->parent == me);
        assert(left->key <= me->key);
        rbt_check_valid(left);
        left_height = rbt_black_height(left);
    }
    size_t right_height = 1;
    if (right) {
        assert(right->parent == me);
        assert(right->key >= me->key);
        rbt_check_valid(right);
        right_height = rbt_black_height(right);
    }
    assert(left_height == right_height);
}
