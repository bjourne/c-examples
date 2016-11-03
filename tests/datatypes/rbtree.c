#include <assert.h>
#include <stdarg.h>
#include <time.h>
#include "datatypes/rbtree.h"
#include "datatypes/vector.h"

rbtree *
add_items(rbtree *t, size_t n, ...) {
    va_list ap;
    va_start(ap, n);
    for (size_t i = 0; i < n; i++) {
        ptr el = va_arg(ap, ptr);
        t = rbt_add(t, el);
    }
    return t;
}

rbtree *
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

    t = rbt_add(t, 20);
    t = rbt_remove(t, rbt_find(t, 20));
    assert(!t);

    t = add_items(NULL, 4, 30, 20, 40, 10);
    n = rbt_find(t, 10);
    assert(n->is_red);
    t = rbt_remove(t, n);
    assert(!t->left->left);
    assert(!t->left->right);
    rbt_free(t);

    t = add_items(NULL, 5, 30, 20, 40, 35, 50);
    t = rbt_remove(t, rbt_find(t, 20));
    assert(t->data == 40);
    rbt_free(t);

    t = add_items(NULL, 4, 30, 20, 40, 35);
    t = rbt_remove(t, rbt_find(t, 20));
    assert(t->data == 35);
    rbt_free(t);

    t = add_items(NULL, 8, 22, 6, 55, 3, 7, 23, 71, 24);
    t = rbt_remove(t, rbt_find(t, 71));

    assert(t->right->data == 24);
    assert(t->right->is_red);
    assert(t->right->right->data == 55);

    rbt_free(t);

    t = add_items(NULL, 8, 74, 41, 76, 34, 62, 25, 63, 72);
    t = rbt_remove(t, rbt_find(t, 76));
    assert(t->data == 41);
    rbt_free(t);

    t = add_items(NULL, 8, 77, 35, 73, 23, 19, 34, 58, 63);
    t = rbt_remove(t, rbt_find(t, 23));
    assert(t->left->data == 34);
    assert(t->data == 35);
    assert(t->left->left->is_red);
    rbt_free(t);

    t = add_items(NULL, 8, 7, 78, 68, 7, 3, 47, 10, 21);
    t = rbt_remove(t, rbt_find(t, 3));
    assert(t->left->right->is_red);
    assert(t->left->right->data == 7);
    rbt_free(t);

    t = add_items(NULL, 8, 8, 69, 0, 74, 71, 76, 51, 74);
    assert(t->data == 8);
    t = rbt_remove(t, rbt_find(t, 0));
    assert(t->data == 71);
    rbt_free(t);
}

void
test_torture() {
    vector *v = v_init(32);
    rbtree *t = NULL;
    size_t value_range = 100000;
    for (int i = 0; i < 100000; i++) {
        if (rand_n(3) == 0 && v->used > 0) {
            size_t i = rand_n(v->used);
            ptr key = v->array[i];
            rbtree *n = rbt_find(t, key);
            t = rbt_remove(t, n);
            v_remove_at(v, i);
        } else {
            ptr key = rand_n(value_range);
            v_add(v, key);
            t = rbt_add(t, key);
        }
    }
    rbt_free(t);
    v_free(v);
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
test_black_height() {
    assert(rbt_black_height(NULL) == 1);
    rbtree *t = rbt_add(NULL, 20);
    assert(rbt_black_height(t) == 2);
    rbt_free(t);
}

int
main(int argc, char *argv[]) {
    time_t seed = time(NULL);
    srand(seed);
    printf("seed %lu\n", seed);
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
    return 0;
}
