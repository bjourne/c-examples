// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "datatypes/queue.h"

void
test_queue() {
    queue *q = queue_init(100, sizeof(int), false);
    int data = 123;
    assert(q->capacity == 100);
    assert(queue_add(q, &data));
    assert(q->n_elements == 1);
    assert(queue_add(q, &data));
    assert(q->n_elements == 2);
    queue_free(q);
}

void
test_queue_fill() {
    queue *q = queue_init(2, sizeof(int), false);
    int v1 = 123, v2 = 321;
    assert(queue_add(q, &v1));
    assert(queue_add(q, &v2));
    assert(!queue_add(q, &v2));
    assert(queue_remove(q, &v1));
    assert(v1 == 123);
    assert(queue_remove(q, &v2));
    assert(v2 == 321);
    assert(q->n_elements == 0);
    queue_free(q);
}

void
test_growing() {
    queue *q = queue_init(2, sizeof(int), true);
    for (int i = 0; i < 10; i++) {
        assert(queue_add(q, &i));
    }
    assert(q->n_elements == 10);
    assert(q->capacity == 13);
    for (int i = 0; i < 10; i++) {
        assert(queue_add(q, &i));
    }
    assert(q->capacity == 28);
    queue_free(q);
}

int
main (int argc, char *argv[]) {
    PRINT_RUN(test_queue);
    PRINT_RUN(test_queue_fill);
    PRINT_RUN(test_growing);
    return 0;
}
