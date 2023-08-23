// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// A supposedly thread-safe queue.
#ifndef SYNCED_QUEUE_H
#define SYNCED_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include "datatypes/queue.h"

typedef struct {
    pthread_spinlock_t lock;
    pthread_mutex_t mutex;
    pthread_cond_t var_prod;
    pthread_cond_t var_cons;
    bool use_spinlocks;
    queue *queue;
} synced_queue;

synced_queue *synced_queue_init(size_t capacity, size_t el_size, bool use_spinlocks);
void synced_queue_free(synced_queue *me);
void synced_queue_add(synced_queue *me, void *value);
void synced_queue_remove(synced_queue *me, void *value);




#endif
