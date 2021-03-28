#include <assert.h>
#include <stdio.h>
#include "datatypes/bstree.h"
#include "datatypes/vector.h"

static bstree *
bst_add_key(bstree *me, bstkey key) {
    return bst_add(me, key, key);
}

void
test_max_min() {
    assert(!bst_iterate(NULL, NULL, BST_RIGHT));
    assert(!bst_iterate(NULL, NULL, BST_LEFT));
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
    assert(!t->childs[BST_LEFT]);
    assert(!t->childs[BST_RIGHT]);
    bst_free(t);

    t = NULL;
    t = bst_add_key(t, 10);
    t = bst_add_key(t, 20);
    t = bst_add_key(t, 30);

    assert(t->key == 10);
    assert(t->childs[BST_RIGHT]->key == 20);
    assert(t->childs[BST_RIGHT]->childs[BST_RIGHT]->key == 30);

    t = bst_remove(t, bst_find(t, 20));
    assert(t->childs[BST_RIGHT]->key == 30);
    t = bst_add_key(t, 2);
    assert(t->childs[BST_LEFT]->key == 2);
    assert(bst_iterate(t, NULL, BST_LEFT)->key == 2);
    assert(bst_iterate(t, NULL, BST_RIGHT)->key == 30);

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
    assert(t == t->childs[BST_LEFT]->parent);
    t = bst_add_key(t, 10);
    assert(t->childs[BST_LEFT] == t->childs[BST_LEFT]->childs[BST_LEFT]->parent);

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
    assert(!t->childs[BST_RIGHT]);

    t = bst_add_key(t, 7);
    t = bst_add_key(t, 5);
    t = bst_remove(t, bst_find(t, 7));
    assert(t->childs[BST_LEFT]->key == 5);
    assert(t->key == 10);
    t = bst_add_key(t, 20);
    assert(t->childs[BST_RIGHT]->key == 20);

    t = bst_remove(t, bst_find(t, 10));
    assert(t->key == 20);
    assert(t->childs[BST_LEFT]->key == 5);

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
    assert(!bst_iterate(t, NULL, BST_LEFT));

    t = bst_add_key(t, 20);
    assert(bst_iterate(t, NULL, BST_LEFT)->key == 20);
    assert(!bst_iterate(t, bst_find(t, 20), BST_LEFT));

    t = bst_add_key(t, 30);
    assert(bst_iterate(t, bst_find(t, 20), BST_LEFT)->key == 30);

    t = bst_add_key(t, 10);

    bstree *succ_1 = bst_iterate(t, NULL, BST_LEFT);
    assert(succ_1->key == 10);
    bstree *succ_2 = bst_iterate(t, succ_1, BST_LEFT);
    assert(succ_2->key == 20);
    bstree *succ_3 = bst_iterate(t, succ_2, BST_LEFT);
    assert(succ_3->key == 30);
    bst_free(t);

    t = NULL;
    for (int i = 0; i < 10; i++) {
        t = bst_add_key(t, rand_n(100));
    }
    bstree *iter = bst_iterate(t, NULL, BST_LEFT);
    int x = 0;
    while (iter) {
        iter = bst_iterate(t, iter, BST_LEFT);
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
        iter = bst_iterate(t, iter, BST_LEFT);
        if (!iter || iter->key != key) {
            break;
        }
        key = iter->key;
    }
    assert(count == 5);
    bst_free(t);
}

void
test_torture() {
    vector *v = v_init(32);
    bstree *t = NULL;
    int range = 1000000;
    int count = 10000000;
    for (int i = 0; i < count; i++) {
        bstkey key = rand_n(range);
        t = bst_add_key(t, key);
        v_add(v, key);
    }
    for (int i = 0; i < count; i++) {
        bstkey key = (bstkey)v->array[i];
        t = bst_remove(t, bst_find(t, key));
    }
    assert(!t);
    bst_free(t);
    v_free(v);
}

// This test is just so that the red-black tree can shine. :)
void
test_unbalancing_torture() {
    bstree *t = NULL;
    for (int i = 0; i < 50000; i++) {
        t = bst_add_key(t, i);
    }
    bst_free(t);
}

void
test_negative_values() {
    bstree *t = NULL;
    t = bst_add_key(t, -5);
    t = bst_add_key(t, -3);
    t = bst_add_key(t, 0);
    t = bst_add_key(t, 8);
    bst_print(t, 0, false);
    bstree *n = bst_iterate(t, NULL, BST_LEFT);
    assert(n->key == -5);
    bst_free(t);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_max_min);
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
    PRINT_RUN(test_torture);
    PRINT_RUN(test_unbalancing_torture);
    PRINT_RUN(test_negative_values);
    return 0;
}
