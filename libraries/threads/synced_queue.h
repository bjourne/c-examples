// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// A supposedly thread-safe queue.
#ifndef SYNCED_QUEUE_H
#define SYNCED_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include "datatypes/queue.h"

typedef enum {
    SYNCED_QUEUE_SPIN_LOCK,
    // This primitive can be used when there is only one producer and
    // one consumer for a queue.
    SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED,
    SYNCED_QUEUE_MUTEX
} synced_queue_lock_type;

typedef struct {
    pthread_spinlock_t lock;
    pthread_mutex_t mutex;
    pthread_cond_t var_prod;
    pthread_cond_t var_cons;
    synced_queue_lock_type lock_type;
    queue *queue;
} synced_queue;

synced_queue *synced_queue_init(size_t capacity, size_t el_size,
                                synced_queue_lock_type lock_type);
void synced_queue_free(synced_queue *me);
void synced_queue_add(synced_queue *me, void *value);
void synced_queue_remove(synced_queue *me, void *value);




#endif
