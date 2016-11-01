#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "datatypes/bstree.h"

void
test_add_remove() {
    bstree *t = NULL;
    t = bst_add(NULL, 123);
    t = bst_remove(t, bst_find(t, 123));
    assert(t == NULL);

    t = bst_add(t, 10);
    t = bst_add(t, 20);
    t = bst_remove(t, bst_find(t, 20));
    assert(t->left == NULL);
    assert(t->right == NULL);
    bst_free(t);

    t = NULL;
    t = bst_add(t, 10);
    t = bst_add(t, 20);
    t = bst_add(t, 30);

    assert(t->data == 10);
    assert(t->right->data == 20);
    assert(t->right->right->data == 30);

    t = bst_remove(t, bst_find(t, 20));
    assert(t->right->data == 30);
    t = bst_add(t, 2);
    assert(t->left->data == 2);
    assert(bst_min_node(t)->data == 2);
    assert(bst_max_node(t)->data == 30);

    t = bst_remove(t, bst_find(t, 10));
    assert(t->data == 30);
    bst_free(t);

    t = NULL;
    t = bst_add(t, 50);
    t = bst_add(t, 40);
    t = bst_add(t, 70);
    t = bst_add(t, 60);
    t = bst_add(t, 80);
    assert(bst_size(t) == 5);

    assert(bst_find(t, 80)->data == 80);
    assert(!bst_find(t, 999));

    t = bst_remove(t, bst_find(t, 50));
    assert(t->data == 60);
    bst_free(t);
}

void
test_callstack_overflow() {
    bstree *bst = NULL;
    size_t count = 10000;
    for (int i = 0; i < count; i++) {
        bst = bst_add(bst, i);
    }
    assert(bst_size(bst) == count);
    bst_free(bst);
}

void
test_print_tree() {
    bstree *bst = NULL;
    for (int i = 0; i < 20; i++) {
        bst = bst_add(bst, rand_n(50));
    }
    bst_print(bst, 0, true);
    bst_free(bst);
}

void
test_more_remove() {
    bstree *bst = NULL;
    bst = bst_add(bst, 222);

    bst = bst_remove(bst, bst_find(bst, 222));
    assert(!bst);
    bst_free(bst);
}

void
test_find_lower_bound() {
    bstree *t = NULL;
    assert(!bst_find_lower_bound(t, 99, NULL));

    t = bst_add(t, 20);
    assert(bst_find_lower_bound(t, 20, NULL)->data == 20);
    assert(!bst_find_lower_bound(t, 30, NULL));
    assert(bst_find_lower_bound(t, 15, NULL)->data == 20);

    t = bst_add(t, 10);
    assert(bst_find_lower_bound(t, 15, NULL)->data == 20);
    assert(bst_find_lower_bound(t, 10, NULL)->data == 10);
    assert(bst_find_lower_bound(t, 19, NULL)->data == 20);

    t = bst_add(t, 30);
    assert(bst_find_lower_bound(t, 29, NULL)->data == 30);
    t = bst_add(t, 40);
    assert(bst_find_lower_bound(t, 39, NULL)->data == 40);
    assert(bst_find_lower_bound(t, 30, NULL)->data == 30);

    t = bst_add(t, 25);
    assert(bst_find_lower_bound(t, 24, NULL)->data == 25);
    assert(bst_find_lower_bound(t, 25, NULL)->data == 25);
    assert(bst_find_lower_bound(t, 26, NULL)->data == 30);
    t = bst_add(t, 12);
    assert(bst_find_lower_bound(t, 11, NULL)->data == 12);

    t = bst_add(t, 17);
    assert(bst_find_lower_bound(t, 16, NULL)->data == 17);
    assert(!bst_find_lower_bound(t, 100, NULL));

    bst_free(t);
}

void
test_parents() {
    bstree *t = bst_add(NULL, 1234);
    assert(!t->parent);
    t = bst_add(t, 22);
    assert(t == t->left->parent);
    t = bst_add(t, 10);
    assert(t->left == t->left->left->parent);

    bstree *n = bst_find(t, 10);
    assert(n->data == 10);
    bst_free(t);
}

void
test_remove_nodes() {
    bstree *t = NULL;

    t = bst_add(t, 10);
    t = bst_add(t, 20);

    t = bst_remove(t, bst_find(t, 20));
    assert(!t->right);

    t = bst_add(t, 7);
    t = bst_add(t, 5);
    t = bst_remove(t, bst_find(t, 7));
    assert(t->left->data == 5);
    assert(t->data == 10);
    t = bst_add(t, 20);
    assert(t->right->data == 20);

    t = bst_remove(t, bst_find(t, 10));
    assert(t->data == 20);
    assert(t->left->data == 5);

    t = bst_remove(t, bst_find(t, 20));
    assert(t->data == 5);
    t = bst_remove(t, bst_find(t, 5));
    assert(!t);

    bst_free(t);
}

void
test_add_remove_torture() {
    bstree *t = NULL;
    int value_range = 10000;
    for (int i = 0; i < 1000000; i++) {
        if (rand_n(2) == 0) {
            bstree *v = bst_find(t, rand_n(value_range));
            if (v) {
                t = bst_remove(t, v);
            }
        } else {
            t = bst_add(t, rand_n(value_range));
        }
    }
    bst_free(t);
}

void
test_successor() {
    bstree *t = NULL;
    assert(!bst_successor(t, NULL));

    t = bst_add(t, 20);
    assert(bst_successor(t, NULL)->data == 20);
    assert(!bst_successor(t, bst_find(t, 20)));

    t = bst_add(t, 30);
    assert(bst_successor(t, bst_find(t, 20))->data == 30);

    t = bst_add(t, 10);

    bstree *succ_1 = bst_successor(t, NULL);
    assert(succ_1->data == 10);
    bstree *succ_2 = bst_successor(t, succ_1);
    assert(succ_2->data == 20);
    bstree *succ_3 = bst_successor(t, succ_2);
    assert(succ_3->data == 30);
    bst_free(t);

    t = NULL;
    for (int i = 0; i < 10; i++) {
        t = bst_add(t, rand_n(100));
    }
    bstree *iter = bst_successor(t, NULL);
    int x = 0;
    while (iter) {
        iter = bst_successor(t, iter);
        x++;
    }
    assert(x == 10);
}

int
main(int argc, char *argv[]) {
    srand(time(NULL));
    PRINT_RUN(test_add_remove);
    PRINT_RUN(test_callstack_overflow);
    PRINT_RUN(test_print_tree);
    PRINT_RUN(test_more_remove);
    PRINT_RUN(test_find_lower_bound);
    PRINT_RUN(test_parents);
    PRINT_RUN(test_remove_nodes);
    PRINT_RUN(test_add_remove_torture);
    PRINT_RUN(test_successor);
    return 0;
}
