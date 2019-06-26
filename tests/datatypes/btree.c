// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include "datatypes/btree.h"

bnode *
create_node(int count, ...) {
    va_list ap;
    va_start(ap, count);
    bnode *node = bnode_init();
    node->count = count;
    for (int i = 0; i < count; i++) {
        int key = va_arg(ap, int);
        node->keys[i] = key;
        node->vals[i] = key;
    }
    return node;
}

void
test_linear_search() {
    bnode *n = bnode_init();
    assert(n->count == 0);

    bool found = false;
    int i;

    n->count = 1;
    n->keys[0] = 30;
    i = bnode_linear_search(n, 10, &found);
    assert(!found && i == 0);
    i = bnode_linear_search(n, 30, &found);
    assert(found && i == 0);

    i = bnode_linear_search(n, 40, &found);
    assert(!found && i == 1);

    n->count = 2;
    n->keys[1] = 50;

    i = bnode_linear_search(n, 40, &found);
    assert(!found && i == 1);

    i = bnode_linear_search(n, 90, &found);
    assert(!found && i == 2);

    bnode_free(n);
}

void
test_find() {
    btree *t = btree_init();

    t->root = create_node(3, 10, 20, 30);

    int i;
    bnode *f;
    f = btree_find(t, 10, &i);
    assert(f && i == 0);

    f = btree_find(t, 20, &i);
    assert(f && i == 1);

    f = btree_find(t, 22, &i);
    assert(!f && i == 2);

    t->root->childs[0] = create_node(2, -8, 0);
    t->root->childs[1] = create_node(3, 14, 16, 18);

    f = btree_find(t, 14, &i);
    assert(f && i == 0);

    f = btree_find(t, -4, &i);
    assert(!f && i == 1);
    f = btree_find(t, -8, &i);
    assert(f && i == 0);

    btree_free(t);
}

void
test_find_duplicates() {
    btree *t = btree_init();
    t->root = create_node(3, 1, 1, 2);

    int i;
    bnode *f;
    f = btree_find(t, 1, &i);
    assert(f && i == 0);
    f = btree_find(t, 2, &i);
    assert(f && i == 2);

    btree_free(t);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_linear_search);
    PRINT_RUN(test_find);
    PRINT_RUN(test_find_duplicates);
    return 0;
}
