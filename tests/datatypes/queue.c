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
    int v1 = 123, v2 = 321, v3 = 999;
    assert(queue_add(q, &v1));
    assert(queue_add(q, &v2));

    assert(!queue_add(q, &v2));
    assert(queue_remove(q, &v1));
    assert(v1 == 123);
    assert(q->n_elements == 1);
    assert(queue_add(q, &v3));
    assert(q->n_elements == 2);

    assert(queue_remove(q, &v2));
    assert(v2 == 321);
    assert(q->n_elements == 1);

    assert(queue_remove(q, &v2));
    assert(v2 == 999);
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

void
test_ranges() {
    queue *q = queue_init(10, sizeof(int), false);
    size_t r0, r1, r2;
    queue_ranges(q, &r0, &r1, &r2);
    assert(r0 == r1 && r1 == r2);

    for (int i = 0; i < 5; i++) {
        assert(queue_add(q, &i));
    }
    queue_ranges(q, &r0, &r1, &r2);
    assert(r0 == 0 && r1 == 5 && r2 == 0);
    int v;
    for (int i = 0; i < 5; i++) {
        assert(queue_remove(q, &v));
    }
    queue_ranges(q, &r0, &r1, &r2);
    assert(r0 == 5 && r1 == 5 && r2 == 0);

    assert(q->head == q->tail);
    for (int i = 0; i < 7; i++) {
        assert(queue_add(q, &i));
    }
    assert(q->n_elements == 7);

    queue_ranges(q, &r0, &r1, &r2);
    assert(r0 == 5);
    assert(r1 == q->capacity + 1);

    for (size_t i = r0; i < r1; i++) {
        printf("%d\n", ((int *)q->array)[i]);
    }
    for (size_t i = 0; i < r2; i++) {
        printf("%d\n", ((int *)q->array)[i]);
    }
    queue_free(q);
}

int
main (int argc, char *argv[]) {
    PRINT_RUN(test_queue);
    PRINT_RUN(test_queue_fill);
    PRINT_RUN(test_growing);
    PRINT_RUN(test_ranges);
    return 0;
}
