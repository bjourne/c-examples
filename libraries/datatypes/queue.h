// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef QUEUE_H
#define QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// Very simple growable queue/ring buffer in C.
typedef struct {
    void *array;
    bool growable;
    size_t capacity;
    size_t el_size;
    size_t n_elements;
    size_t head;
    size_t tail;
} queue;

queue* queue_init(size_t capacity, size_t el_size, bool growable);
bool queue_add(queue *me, void *src);
bool queue_remove(queue *me, void *dst);
void queue_free(queue *me);

#endif
