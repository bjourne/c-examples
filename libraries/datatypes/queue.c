// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <string.h>
#include "queue.h"

queue*
queue_init(size_t capacity, size_t el_size, bool growable) {
    queue *me = malloc(sizeof(queue));
    me->array = malloc(el_size * capacity);
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
        me->array = realloc(me->array, me->capacity * me->el_size);
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

void
queue_free(queue *me) {
    free(me->array);
    free(me);
}
