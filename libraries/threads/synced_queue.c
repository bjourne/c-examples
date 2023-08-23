// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include "threads/synced_queue.h"

synced_queue *
synced_queue_init(size_t capacity, size_t el_size, bool use_spinlocks) {
    synced_queue *me = malloc(sizeof(synced_queue));
    me->queue = queue_init(capacity, el_size, false);
    me->use_spinlocks = use_spinlocks;
    if (me->use_spinlocks) {
        assert(!pthread_spin_init(&me->lock, PTHREAD_PROCESS_PRIVATE));
    } else {
        assert(!pthread_mutex_init(&me->mutex, NULL));
        assert(!pthread_cond_init(&me->var_prod, NULL));
        assert(!pthread_cond_init(&me->var_cons, NULL));
    }
    return me;
}

void
synced_queue_free(synced_queue *me) {
    if (me->use_spinlocks) {
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
        if (me->use_spinlocks) {
            spin_while_full(me);
            pthread_spin_lock(&me->lock);
        } else {
            pthread_mutex_lock(&me->mutex);
            while (me->queue->n_elements == me->queue->capacity) {
                pthread_cond_wait(&me->var_prod, &me->mutex);
            }
        }
        queue_add(me->queue, value);
        if (me->use_spinlocks) {
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
        if (me->use_spinlocks) {
            spin_while_empty(me);
            pthread_spin_lock(&me->lock);
        } else {
            pthread_mutex_lock(&me->mutex);
            while (me->queue->n_elements == 0) {
                pthread_cond_wait(&me->var_cons, &me->mutex);
            }
        }
        queue_remove(me->queue, value);
        if (me->use_spinlocks) {
            pthread_spin_unlock(&me->lock);
        } else {
            pthread_mutex_unlock(&me->mutex);
            pthread_cond_signal(&me->var_prod);
        }
    }
