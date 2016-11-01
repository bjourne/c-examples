#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "datatypes/bstree.h"

static void
test_add_remove() {
    bstree *bst1 = bst_add(NULL, 123);
    bst1 = bst_remove(bst1, 123);
    assert(bst1 == NULL);

    bstree *bst2 = NULL;
    bst2 = bst_add(bst2, 10);
    bst2 = bst_add(bst2, 20);
    bst2 = bst_remove(bst2, 20);
    assert(bst2->left == NULL);
    assert(bst2->right == NULL);
    bst_free(bst2);

    bstree *bst3 = NULL;
    bst_remove(bst3, 30);

    bst3 = bst_add(bst3, 10);
    bst3 = bst_add(bst3, 20);
    bst3 = bst_add(bst3, 30);

    assert(bst3->data == 10);
    assert(bst3->right->data == 20);
    assert(bst3->right->right->data == 30);

    bst3 = bst_remove(bst3, 20);
    assert(bst3->right->data == 30);
    bst3 = bst_add(bst3, 2);
    assert(bst3->left->data == 2);
    assert(bst_min_node(bst3)->data == 2);
    assert(bst_max_node(bst3)->data == 30);

    bst3 = bst_remove(bst3, 10);
    assert(bst3->data == 30);
    bst_free(bst3);


    bstree *bst4 = NULL;
    bst4 = bst_add(bst4, 50);
    bst4 = bst_add(bst4, 40);
    bst4 = bst_add(bst4, 70);
    bst4 = bst_add(bst4, 60);
    bst4 = bst_add(bst4, 80);
    assert(bst_size(bst4) == 5);

    assert(bst_find(bst4, 80)->data == 80);

    bst4 = bst_remove(bst4, 50);
    bst_free(bst4);
}

static void
test_callstack_overflow() {
    bstree *bst = NULL;
    size_t count = 10000;
    for (int i = 0; i < count; i++) {
        bst = bst_add(bst, i);
    }
    assert(bst_size(bst) == count);
    bst_free(bst);
}

static void
test_print_tree() {
    bstree *bst = NULL;
    for (int i = 0; i < 20; i++) {
        bst = bst_add(bst, rand_n(50));
    }
    bst_print(bst, 0, false);
    bst_free(bst);
}

void
test_more_remove() {
    bstree *bst = NULL;
    bst = bst_add(bst, 222);

    assert(bst_remove(bst, 9));

    bst = bst_remove(bst, 222);
    assert(!bst);
    bst_free(bst);
}

void
test_find_lower_bound() {
    bstree *t = NULL;
    assert(!bst_find_lower_bound(t, 99));

    t = bst_add(t, 20);
    assert(bst_find_lower_bound(t, 20)->data == 20);
    assert(!bst_find_lower_bound(t, 30));
    assert(bst_find_lower_bound(t, 15)->data == 20);

    t = bst_add(t, 10);
    assert(bst_find_lower_bound(t, 15)->data == 20);
    assert(bst_find_lower_bound(t, 10)->data == 10);
    assert(bst_find_lower_bound(t, 19)->data == 20);

    t = bst_add(t, 30);
    assert(bst_find_lower_bound(t, 29)->data == 30);
    t = bst_add(t, 40);
    assert(bst_find_lower_bound(t, 39)->data == 40);
    assert(bst_find_lower_bound(t, 30)->data == 30);

    t = bst_add(t, 25);
    assert(bst_find_lower_bound(t, 24)->data == 25);
    assert(bst_find_lower_bound(t, 25)->data == 25);
    assert(bst_find_lower_bound(t, 26)->data == 30);
    t = bst_add(t, 12);
    assert(bst_find_lower_bound(t, 11)->data == 12);

    t = bst_add(t, 17);
    assert(bst_find_lower_bound(t, 16)->data == 17);

    bst_free(t);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_add_remove);
    PRINT_RUN(test_callstack_overflow);
    PRINT_RUN(test_print_tree);
    PRINT_RUN(test_more_remove);
    PRINT_RUN(test_find_lower_bound);
    return 0;
}
