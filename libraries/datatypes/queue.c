// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <string.h>
#include "common.h"
#include "queue.h"

queue*
queue_init(size_t capacity, size_t el_size, bool growable) {
    queue *me = malloc(sizeof(queue));
    size_t n_bytes = el_size * (capacity + 1);
    me->array = malloc(n_bytes);
    me->growable = growable;
    me->capacity = capacity;
    me->el_size = el_size;
    me->head = 0;
    me->tail = 0;
    me->n_elements = 0;
    return me;
}

bool
queue_add(queue *me, void *src) {
    if (me->n_elements == me->capacity) {
        return false;
    }
    size_t next = (me->head + 1) % (me->capacity + 1);
    memcpy(me->array + me->el_size * me->head, src, me->el_size);
    me->head = next;
    me->n_elements++;
    if (me->n_elements == me->capacity && me->growable) {
        me->capacity = me->capacity + me->capacity / 2;
        size_t n_bytes = me->el_size * (me->capacity + 1);
        me->array = realloc(me->array, n_bytes);
    }
    return true;
}

bool
queue_remove(queue *me, void *dst) {
    if (me->n_elements == 0) {
        return false;
    }
    size_t next = (me->tail + 1) % (me->capacity + 1);
    memcpy(dst, me->array + me->el_size * me->tail, me->el_size);
    me->tail = next;
    me->n_elements--;
    return true;
}

// Get indices so that [r0, r1) and [0, r2) iterates the queue in FIFO
// order. This function is more convenient than modulo arithmetic.
void
queue_ranges(queue *me, size_t *r0, size_t *r1, size_t *r2) {
    *r0 = me->tail;
    if (me->head >= me->tail) {
        *r1 = me->head;
        *r2 = 0;
    } else {
        *r1 = me->capacity + 1;
        *r2 = me->head;
    }
}

void
queue_free(queue *me) {
    free(me->array);
    free(me);
}
