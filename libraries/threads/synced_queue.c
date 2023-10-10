// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/common.h"
#include "threads/synced_queue.h"

synced_queue *
synced_queue_init(size_t capacity, size_t el_size,
                  synced_queue_lock_type lock_type) {
    synced_queue *me = malloc_aligned(16, sizeof(synced_queue));
    me->queue = queue_init(capacity, el_size, false);
    me->lock_type = lock_type;
    if (me->lock_type == SYNCED_QUEUE_SPIN_LOCK ||
        me->lock_type == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
        assert(!pthread_spin_init(&me->lock, PTHREAD_PROCESS_PRIVATE));
    } else if (me->lock_type == SYNCED_QUEUE_MUTEX) {
        assert(!pthread_mutex_init(&me->mutex, NULL));
        assert(!pthread_cond_init(&me->var_prod, NULL));
        assert(!pthread_cond_init(&me->var_cons, NULL));
    }
    return me;
}

void
synced_queue_free(synced_queue *me) {
    if (me->lock_type == SYNCED_QUEUE_SPIN_LOCK ||
        me->lock_type == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
        assert(!pthread_spin_destroy(&me->lock));
    } else {
        assert(!pthread_mutex_destroy(&me->mutex));
        assert(!pthread_cond_destroy(&me->var_prod));
        assert(!pthread_cond_destroy(&me->var_cons));
    }
    queue_free(me->queue);
    free(me);
}

static void
spin_while_full(volatile synced_queue *me) {
    while (me->queue->n_elements == me->queue->capacity) {
    }
}

void
synced_queue_add(synced_queue *me, void *value) {
    synced_queue_lock_type tp = me->lock_type;
    if (tp == SYNCED_QUEUE_SPIN_LOCK) {
        while (true) {
            pthread_spin_lock(&me->lock);
            if (me->queue->n_elements < me->queue->capacity) {
                break;
            }
            pthread_spin_unlock(&me->lock);
        }
    } else if (tp == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
        spin_while_full(me);
        pthread_spin_lock(&me->lock);
    } else if (tp == SYNCED_QUEUE_MUTEX) {
        pthread_mutex_lock(&me->mutex);
        while (me->queue->n_elements == me->queue->capacity) {
            pthread_cond_wait(&me->var_prod, &me->mutex);
        }
    } else {
        assert(false);
    }
    queue_add(me->queue, value);
    if (tp == SYNCED_QUEUE_SPIN_LOCK ||
        tp == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
        pthread_spin_unlock(&me->lock);
    } else {
        pthread_mutex_unlock(&me->mutex);
        pthread_cond_signal(&me->var_cons);
    }
}

static void
spin_while_empty(volatile synced_queue *me) {
    while (me->queue->n_elements == 0) {
    }
}

void
synced_queue_remove(synced_queue *me, void *value) {
    synced_queue_lock_type tp = me->lock_type;
    if (tp == SYNCED_QUEUE_SPIN_LOCK) {
        while (true) {
            pthread_spin_lock(&me->lock);
            if (me->queue->n_elements > 0) {
                break;
            }
            pthread_spin_unlock(&me->lock);
        }
    } else if (tp == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
        spin_while_empty(me);
        pthread_spin_lock(&me->lock);
    } else if (tp == SYNCED_QUEUE_MUTEX) {
        pthread_mutex_lock(&me->mutex);
        while (me->queue->n_elements == 0) {
            pthread_cond_wait(&me->var_cons, &me->mutex);
        }
    } else {
        assert(false);
    }
    queue_remove(me->queue, value);
    if (tp == SYNCED_QUEUE_SPIN_LOCK ||
        tp == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
        pthread_spin_unlock(&me->lock);
    } else {
        pthread_mutex_unlock(&me->mutex);
        pthread_cond_signal(&me->var_prod);
    }
}
