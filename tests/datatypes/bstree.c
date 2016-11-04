#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "datatypes/bstree.h"

static bstree *
bst_add_key(bstree *me, size_t key) {
    return bst_add(me, key, key);
}

void
test_add_remove() {
    bstree *t = NULL;
    t = bst_add_key(NULL, 123);
    t = bst_remove(t, bst_find(t, 123));
    assert(t == NULL);

    t = bst_add_key(t, 10);
    t = bst_add_key(t, 20);
    t = bst_remove(t, bst_find(t, 20));
    assert(t->left == NULL);
    assert(t->right == NULL);
    bst_free(t);

    t = NULL;
    t = bst_add_key(t, 10);
    t = bst_add_key(t, 20);
    t = bst_add_key(t, 30);

    assert(t->key == 10);
    assert(t->right->key == 20);
    assert(t->right->right->key == 30);

    t = bst_remove(t, bst_find(t, 20));
    assert(t->right->key == 30);
    t = bst_add_key(t, 2);
    assert(t->left->key == 2);
    assert(bst_min_node(t)->key == 2);
    assert(bst_max_node(t)->key == 30);

    t = bst_remove(t, bst_find(t, 10));
    assert(t->key == 30);
    bst_free(t);

    t = NULL;
    t = bst_add_key(t, 50);
    t = bst_add_key(t, 40);
    t = bst_add_key(t, 70);
    t = bst_add_key(t, 60);
    t = bst_add_key(t, 80);
    assert(bst_size(t) == 5);

    assert(bst_find(t, 80)->key == 80);
    assert(!bst_find(t, 999));

    t = bst_remove(t, bst_find(t, 50));
    assert(t->key == 60);
    bst_free(t);
}

void
test_callstack_overflow() {
    bstree *bst = NULL;
    size_t count = 10000;
    for (int i = 0; i < count; i++) {
        bst = bst_add_key(bst, i);
    }
    assert(bst_size(bst) == count);
    bst_free(bst);
}

void
test_print_tree() {
    bstree *bst = NULL;
    for (int i = 0; i < 20; i++) {
        bst = bst_add_key(bst, rand_n(50));
    }
    bst_print(bst, 0, true);
    bst_free(bst);
}

void
test_more_remove() {
    bstree *bst = NULL;
    bst = bst_add_key(bst, 222);

    bst = bst_remove(bst, bst_find(bst, 222));
    assert(!bst);
    bst_free(bst);
}

void
test_find_lower_bound() {
    bstree *t = NULL;
    assert(!bst_find_lower_bound(t, 99));

    t = bst_add_key(t, 20);
    assert(bst_find_lower_bound(t, 20)->key == 20);
    assert(!bst_find_lower_bound(t, 30));
    assert(bst_find_lower_bound(t, 15)->key == 20);

    t = bst_add_key(t, 10);
    assert(bst_find_lower_bound(t, 15)->key == 20);
    assert(bst_find_lower_bound(t, 10)->key == 10);
    assert(bst_find_lower_bound(t, 19)->key == 20);

    t = bst_add_key(t, 30);
    assert(bst_find_lower_bound(t, 29)->key == 30);
    t = bst_add_key(t, 40);
    assert(bst_find_lower_bound(t, 39)->key == 40);
    assert(bst_find_lower_bound(t, 30)->key == 30);

    t = bst_add_key(t, 25);
    assert(bst_find_lower_bound(t, 24)->key == 25);
    assert(bst_find_lower_bound(t, 25)->key == 25);
    assert(bst_find_lower_bound(t, 26)->key == 30);
    t = bst_add_key(t, 12);
    assert(bst_find_lower_bound(t, 11)->key == 12);

    t = bst_add_key(t, 17);
    assert(bst_find_lower_bound(t, 16)->key == 17);
    assert(!bst_find_lower_bound(t, 100));

    bst_free(t);
}

void
test_parents() {
    bstree *t = bst_add_key(NULL, 1234);
    assert(!t->parent);
    t = bst_add_key(t, 22);
    assert(t == t->left->parent);
    t = bst_add_key(t, 10);
    assert(t->left == t->left->left->parent);

    bstree *n = bst_find(t, 10);
    assert(n->key == 10);
    bst_free(t);
}

void
test_remove_nodes() {
    bstree *t = NULL;

    t = bst_add_key(t, 10);
    t = bst_add_key(t, 20);

    t = bst_remove(t, bst_find(t, 20));
    assert(!t->right);

    t = bst_add_key(t, 7);
    t = bst_add_key(t, 5);
    t = bst_remove(t, bst_find(t, 7));
    assert(t->left->key == 5);
    assert(t->key == 10);
    t = bst_add_key(t, 20);
    assert(t->right->key == 20);

    t = bst_remove(t, bst_find(t, 10));
    assert(t->key == 20);
    assert(t->left->key == 5);

    t = bst_remove(t, bst_find(t, 20));
    assert(t->key == 5);
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
            t = bst_add_key(t, rand_n(value_range));
        }
    }
    bst_free(t);
}

void
test_successor() {
    bstree *t = NULL;
    assert(!bst_successor(t, NULL));

    t = bst_add_key(t, 20);
    assert(bst_successor(t, NULL)->key == 20);
    assert(!bst_successor(t, bst_find(t, 20)));

    t = bst_add_key(t, 30);
    assert(bst_successor(t, bst_find(t, 20))->key == 30);

    t = bst_add_key(t, 10);

    bstree *succ_1 = bst_successor(t, NULL);
    assert(succ_1->key == 10);
    bstree *succ_2 = bst_successor(t, succ_1);
    assert(succ_2->key == 20);
    bstree *succ_3 = bst_successor(t, succ_2);
    assert(succ_3->key == 30);
    bst_free(t);

    t = NULL;
    for (int i = 0; i < 10; i++) {
        t = bst_add_key(t, rand_n(100));
    }
    bstree *iter = bst_successor(t, NULL);
    int x = 0;
    while (iter) {
        iter = bst_successor(t, iter);
        x++;
    }
    assert(x == 10);
}

// Here we are trying to find all nodes with the same data.
void
test_successors_with_duplicates() {
    bstree *t = NULL;
    t = bst_add_key(t, 20);
    t = bst_add_key(t, 30);
    t = bst_add_key(t, 77);
    t = bst_add_key(t, 30);
    t = bst_add_key(t, 30);
    t = bst_add_key(t, 30);
    t = bst_add_key(t, 30);
    t = bst_add_key(t, 99);
    t = bst_remove(t, bst_find(t, 20));
    bst_print(t, 0, true);

    bstree *iter = bst_find(t, 30);
    size_t key = iter->key;
    size_t count = 0;
    while (true) {
        count++;
        iter = bst_successor(t, iter);
        if (!iter || iter->key != key) {
            break;
        }
        key = iter->key;
    }
    assert(count == 5);
    bst_free(t);
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
    PRINT_RUN(test_successors_with_duplicates);
    return 0;
}
