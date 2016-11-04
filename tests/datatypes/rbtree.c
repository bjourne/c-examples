#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include "datatypes/rbtree.h"
#include "datatypes/vector.h"

static rbtree *
rbt_add_key(rbtree *me, ptr key) {
    return rbt_add(me, key, key);
}

static rbtree *
add_items(rbtree *t, size_t n, ...) {
    va_list ap;
    va_start(ap, n);
    for (size_t i = 0; i < n; i++) {
        ptr el = va_arg(ap, ptr);
        t = rbt_add_key(t, el);
    }
    return t;
}

static rbtree *
remove_items(rbtree *t, size_t n, ...) {
    va_list ap;
    va_start(ap, n);
    for (size_t i = 0; i < n; i++) {
        ptr el = va_arg(ap, ptr);
        rbtree *n = rbt_find(t, el);
        t = rbt_remove(t, n);
    }
    return t;
}

rbtree *
traverse(rbtree *t, const char *path) {
    for (size_t i = 0; i < strlen(path); i++) {
        bstdir dir = path[i] == 'R' ? BST_RIGHT : BST_LEFT;
        t = t->childs[dir];
    }
    return t;
}

void
test_max_min() {
    rbtree *t = NULL;
    t = add_items(t, 5, 10, 99, 30, 17, 29);

    assert(rbt_iterate(t, NULL, BST_LEFT)->key == 10);
    assert(rbt_iterate(t, NULL, BST_RIGHT)->key == 99);
    rbt_free(t);
}

void
test_many_duplicates() {
    rbtree *t = NULL;
    rbtree *n = NULL;
    t = rbt_add_key(t, 5);
    t = rbt_add_key(t, 1);
    t = rbt_add_key(t, 7);
    t = rbt_add_key(t, 1);
    t = rbt_add_key(t, 3);
    t = rbt_add_key(t, 5);
    t = rbt_add_key(t, 7);
    t = rbt_add_key(t, 1);
    t = rbt_add_key(t, 5);
    t = rbt_add_key(t, 9);
    rbt_print(t, 0, false);

    assert(!rbt_find(t, 102));

    // Find 3 ones
    n = rbt_find(t, 1);
    assert(n && n->key == 1);
    t = rbt_remove(t, n);

    n = rbt_find(t, 1);
    assert(n && n->key == 1);
    t = rbt_remove(t, n);

    n = rbt_find(t, 1);
    assert(n && n->key == 1);
    t = rbt_remove(t, n);

    assert(!rbt_find(t, 1));

    rbt_free(t);
}

void
test_black_height() {
    assert(rbt_black_height(NULL) == 1);
    rbtree *t = rbt_add_key(NULL, 20);
    assert(rbt_black_height(t) == 2);
    rbt_free(t);
}

void
test_simple() {
    rbtree *t = rbt_add_key(NULL, 123);
    t = rbt_add_key(t, 22);
    t = rbt_add_key(t, 300);

    assert(!t->is_red);
    assert(t->childs[BST_LEFT]->is_red);
    assert(t->childs[BST_RIGHT]->is_red);

    assert(t->parent == NULL);
    assert(t->childs[BST_LEFT]->parent == t);
    assert(t->childs[BST_RIGHT]->parent = t);

    t = rbt_add_key(t, 1);
    assert(!t->is_red);
    assert(!t->childs[BST_LEFT]->is_red);
    assert(t->childs[BST_LEFT]->childs[BST_LEFT]->is_red);
    assert(!t->childs[BST_RIGHT]->is_red);

    rbt_free(t);
}

void
test_simple_2() {
    rbtree *t = rbt_add_key(NULL, 0);
    assert(t->key == 0);
    t = rbt_add_key(t, 1);

    assert(t->key == 0);
    assert(traverse(t, "R")->key == 1);

    t = rbt_add_key(t, 2);
    assert(t->key == 1);
    assert(traverse(t, "L")->key == 0);
    assert(traverse(t, "R")->key == 2);
    t = rbt_add_key(t, 3);

    assert(!traverse(t, "L")->is_red);
    assert(traverse(t, "RR")->is_red);

    assert(traverse(t, "RR")->key == 3);
    t = rbt_add_key(t, 4);
    assert(traverse(t, "R")->key == 3);
    assert(traverse(t, "RL")->key == 2);
    assert(traverse(t, "RR")->key == 4);
    assert(traverse(t, "RL")->is_red);
    assert(traverse(t, "RR")->is_red);
    t = rbt_add_key(t, 5);
    t = rbt_add_key(t, 6);
    assert(traverse(t, "RR")->key == 5);
    assert(traverse(t, "RRR")->key == 6);
    t = rbt_add_key(t, 7);
    assert(t->key == 3);
    assert(traverse(t, "L")->key == 1);
    assert(traverse(t, "LL")->key == 0);
    assert(traverse(t, "LR")->key == 2);
    assert(traverse(t, "R")->key == 5);
    assert(traverse(t, "RL")->key == 4);
    assert(traverse(t, "RR")->key == 6);
    assert(traverse(t, "RRR")->key == 7);
    rbt_free(t);
}

void
test_rotate_left() {
    rbtree *t = rbt_add_key(NULL, 0);
    t = rbt_add_key(t, 10);
    t = rbt_add_key(t, 20);
    assert(t->key == 10);
    assert(t->is_red == false);
    assert(t->childs[BST_LEFT]->key == 0);
    assert(t->childs[BST_RIGHT]->key == 20);
    rbt_free(t);
}

void
test_rotate_right() {
    rbtree *t = rbt_add_key(NULL, 100);
    t = rbt_add_key(t, 90);
    t = rbt_add_key(t, 80);
    assert(t->key == 90);
    rbt_free(t);
}

void
test_double_rotations() {
    rbtree *t = rbt_add_key(NULL, 50);
    t = rbt_add_key(t, 40);
    t = rbt_add_key(t, 45);
    assert(t->key == 45);
    assert(!t->is_red);
    assert(t->childs[BST_LEFT]->key == 40);
    assert(t->childs[BST_LEFT]->is_red);
    assert(t->childs[BST_RIGHT]->key == 50);
    assert(t->childs[BST_RIGHT]->is_red);
    rbt_free(t);

    t = rbt_add_key(NULL, 50);
    t = rbt_add_key(t, 60);
    t = rbt_add_key(t, 55);
    assert(t->key == 55);
    rbt_free(t);
}

void
test_print_tree() {
    rbtree *t = NULL;
    for (int i = 0; i < 10; i++) {
        t = rbt_add_key(t, rand_n(10));
    }
    rbt_print(t, 0, false);
    rbt_free(t);
}

void
test_find() {
    rbtree *t = NULL;
    assert(!rbt_find(t, 99));
    rbt_free(t);
}

void
test_remove() {
    rbtree *t = NULL;
    rbtree *n = NULL;

    t = rbt_add_key(t, 20);
    t = rbt_remove(t, rbt_find(t, 20));
    assert(!t);

    t = add_items(NULL, 4, 30, 20, 40, 10);
    n = rbt_find(t, 10);
    assert(n->is_red);
    t = rbt_remove(t, n);
    assert(!t->childs[BST_LEFT]->childs[BST_LEFT]);
    assert(!t->childs[BST_LEFT]->childs[BST_RIGHT]);
    rbt_free(t);

    t = add_items(NULL, 5, 30, 20, 40, 35, 50);
    t = rbt_remove(t, rbt_find(t, 20));
    assert(t->key == 40);
    rbt_free(t);

    t = add_items(NULL, 4, 30, 20, 40, 35);
    t = rbt_remove(t, rbt_find(t, 20));
    assert(t->key == 35);
    rbt_free(t);

    t = add_items(NULL, 8, 22, 6, 55, 3, 7, 23, 71, 24);
    t = rbt_remove(t, rbt_find(t, 71));

    assert(t->childs[BST_RIGHT]->key == 24);
    assert(t->childs[BST_RIGHT]->is_red);
    assert(t->childs[BST_RIGHT]->childs[BST_RIGHT]->key == 55);

    rbt_free(t);

    t = add_items(NULL, 8, 74, 41, 76, 34, 62, 25, 63, 72);
    t = rbt_remove(t, rbt_find(t, 76));
    assert(t->key == 41);
    rbt_free(t);

    t = add_items(NULL, 8, 77, 35, 73, 23, 19, 34, 58, 63);
    t = rbt_remove(t, rbt_find(t, 23));
    assert(t->childs[BST_LEFT]->key == 34);
    assert(t->key == 35);
    assert(t->childs[BST_LEFT]->childs[BST_LEFT]->is_red);
    rbt_free(t);

    t = add_items(NULL, 8, 7, 78, 68, 7, 3, 47, 10, 21);
    t = rbt_remove(t, rbt_find(t, 3));
    assert(t->childs[BST_LEFT]->childs[BST_RIGHT]->is_red);
    assert(t->childs[BST_LEFT]->childs[BST_RIGHT]->key == 7);
    rbt_free(t);

    t = add_items(NULL, 8, 8, 69, 0, 74, 71, 76, 51, 74);
    assert(t->key == 8);
    t = rbt_remove(t, rbt_find(t, 0));
    assert(t->key == 71);
    rbt_free(t);
}

// 28.5
void
test_torture() {
    vector *v = v_init(32);
    rbtree *t = NULL;
    uint64_t range = 10 * 1000 * 1000 * 1000LL;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t key = rand_n(range);
        t = rbt_add_key(t, key);
        v_add(v, key);
    }
    for (uint64_t i = 0; i < count; i++) {
        uint64_t key = v->array[i];
        t = rbt_remove(t, rbt_find(t, key));
    }
    assert(!t);
    v_free(v);
}

// 12.6
void
test_torture_comp() {
    rbtree *t = NULL;
    uint64_t range = 10 * 1000 * 1000 * 1000LL;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t key = rand_n(range);
        t = rbt_add(t, key, key);
    }
    rbt_free(t);
}

void
test_loop_scenario() {
    rbtree *t = add_items(NULL, 9,
                          271, 952, 36, 893, 437, 232, 905, 2, 816);
    t = remove_items(t, 6,
                     816, 36, 2, 905, 893, 437);
    t = add_items(t, 1, 870);
    t = remove_items(t, 2, 271, 952);
    t = add_items(t, 1, 2);
    t = remove_items(t, 1, 870);
    t = add_items(t, 1, 712);
    t = remove_items(t, 2, 2, 712);
    t = add_items(t, 3, 813, 398, 177);
    t = remove_items(t, 1, 398);
    t = add_items(t, 5, 127, 680, 499, 555, 700);
    t = remove_items(t, 1, 232);
    t = add_items(t, 2, 695, 380);
    t = remove_items(t, 3, 499, 695, 127);
    t = add_items(t, 1, 647);
    t = remove_items(t, 2, 700, 680);
    t = add_items(t, 2, 621, 237);
    t = remove_items(t, 2, 555, 177);
    t = add_items(t, 1, 147);
    t = remove_items(t, 1, 237);
    t = add_items(t, 2, 593, 117);
    t = remove_items(t, 1, 380);
    t = add_items(t, 1, 3);
    t = remove_items(t, 1, 647);
    t = add_items(t, 1, 595);
    t = remove_items(t, 1, 593);
    t = add_items(t, 3, 878, 639, 318);
    t = remove_items(t, 1, 878);
    t = add_items(t, 4, 361, 176, 931, 427);
    t = remove_items(t, 1, 318);
    t = add_items(t, 4, 764, 13, 841, 279);
    t = remove_items(t, 2, 813, 147);
    t = add_items(t, 3, 116, 835, 709);
    t = remove_items(t, 2, 621, 595);
    t = add_items(t, 5, 479, 198, 616, 514, 703, 187);
    t = remove_items(t, 1, 764);
    rbt_free(t);
}

void
test_duplicates_and_rotates() {
    rbtree *t = add_items(NULL, 3, 5, 5, 7);
    rbt_print(t, 0, true);

    // Damn
    rbtree *n1 = rbt_iterate(t, NULL, BST_LEFT);
    assert(n1->key == 5);
    assert(!traverse(n1, "R"));
    rbtree *n2 = rbt_iterate(t, n1, BST_LEFT);
    assert(n2->key == 5);
    rbtree *n3 = rbt_iterate(t, n2, BST_LEFT);
    assert(n3->key == 7);
    rbt_free(t);
}

int
main(int argc, char *argv[]) {
    time_t seed = time(NULL);
    srand(seed);
    PRINT_RUN(test_max_min);
    PRINT_RUN(test_many_duplicates);
    PRINT_RUN(test_black_height);
    PRINT_RUN(test_loop_scenario);
    PRINT_RUN(test_simple);
    PRINT_RUN(test_simple_2);
    PRINT_RUN(test_rotate_left);
    PRINT_RUN(test_rotate_right);
    PRINT_RUN(test_double_rotations);
    PRINT_RUN(test_print_tree);
    PRINT_RUN(test_find);
    PRINT_RUN(test_remove);
    PRINT_RUN(test_torture);
    PRINT_RUN(test_duplicates_and_rotates);
    PRINT_RUN(test_torture_comp);
    return 0;
}
