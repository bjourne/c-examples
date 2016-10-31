#include <assert.h>
#include "datatypes/rbtree.h"

void
test_simple() {
    rbtree *t = rbt_add(NULL, 123);
    t = rbt_add(t, 22);
    t = rbt_add(t, 300);

    assert(!t->is_red);
    assert(t->left->is_red);
    assert(t->right->is_red);

    assert(t->parent == NULL);
    assert(t->left->parent == t);
    assert(t->right->parent = t);

    t = rbt_add(t, 1);
    assert(!t->is_red);
    assert(!t->left->is_red);
    assert(t->left->left->is_red);
    assert(!t->right->is_red);

    rbt_free(t);
}

void
test_simple_2() {
    rbtree *t = rbt_add(NULL, 0);
    assert(t->data == 0);
    t = rbt_add(t, 1);


    assert(t->data == 0);
    assert(t->right->data == 1);

    t = rbt_add(t, 2);
    assert(t->data == 1);
    assert(t->left->data == 0);
    assert(t->right->data == 2);
    t = rbt_add(t, 3);
    assert(t->right->right->data == 3);
    t = rbt_add(t, 4);
    assert(t->right->data == 3);
    assert(t->right->left->data == 2);
    assert(t->right->right->data == 4);
    assert(t->right->left->is_red);
    assert(t->right->right->is_red);
    t = rbt_add(t, 5);
    t = rbt_add(t, 6);
    assert(t->right->right->data == 5);
    assert(t->right->right->right->data == 6);
    t = rbt_add(t, 7);
    assert(t->data == 3);
    assert(t->left->data == 1);
    assert(t->left->left->data == 0);
    assert(t->left->right->data == 2);
    assert(t->right->data == 5);
    assert(t->right->left->data == 4);
    assert(t->right->right->data == 6);
    assert(t->right->right->right->data == 7);
    rbt_free(t);
}

void
test_rotate_left() {
    rbtree *t = rbt_add(NULL, 0);
    t = rbt_add(t, 10);
    t = rbt_add(t, 20);
    assert(t->data == 10);
    assert(t->is_red == false);
    assert(t->left->data == 0);
    assert(t->right->data == 20);
    rbt_free(t);
}

void
test_rotate_right() {
    rbtree *t = rbt_add(NULL, 100);
    t = rbt_add(t, 90);
    t = rbt_add(t, 80);
    assert(t->data == 90);
    rbt_free(t);
}

void
test_double_rotations() {
    rbtree *t = rbt_add(NULL, 50);
    t = rbt_add(t, 40);
    t = rbt_add(t, 45);
    assert(t->data == 45);
    assert(!t->is_red);
    assert(t->left->data == 40);
    assert(t->left->is_red);
    assert(t->right->data == 50);
    assert(t->right->is_red);
    rbt_free(t);

    t = rbt_add(NULL, 50);
    t = rbt_add(t, 60);
    t = rbt_add(t, 55);
    assert(t->data == 55);
    rbt_free(t);
}

void
test_print_tree() {
    rbtree *t = NULL;
    for (int i = 0; i < 10; i++) {
        t = rbt_add(t, rand_n(10));
    }
    rbt_print(t, 0, false);
    rbt_free(t);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_simple);
    PRINT_RUN(test_simple_2);
    PRINT_RUN(test_rotate_left);
    PRINT_RUN(test_rotate_right);
    PRINT_RUN(test_double_rotations);
    PRINT_RUN(test_print_tree);
    return 0;
}
