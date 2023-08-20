// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates the producer-consumer pattern using my threads
// library.
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include "datatypes/queue.h"
#include "random/random.h"
#include "threads/threads.h"

typedef struct {
    queue *q;
    pthread_mutex_t m;
    pthread_cond_t var_prod;
    pthread_cond_t var_cons;
} synced_queue;

synced_queue *
synced_queue_init(size_t max, size_t el_size) {
    synced_queue *me = malloc(sizeof(synced_queue));
    me->q = queue_init(max, el_size, false);
    assert(!pthread_mutex_init(&me->m, NULL));
    assert(!pthread_cond_init(&me->var_prod, NULL));
    assert(!pthread_cond_init(&me->var_cons, NULL));
    return me;
}

void
synced_queue_free(synced_queue *me) {
    assert(!pthread_mutex_destroy(&me->m));
    assert(!pthread_cond_destroy(&me->var_prod));
    assert(!pthread_cond_destroy(&me->var_cons));
    queue_free(me->q);
    free(me);
}

static void
synced_queue_add(synced_queue *me, void *value) {
    pthread_mutex_lock(&me->m);
    while (me->q->n_elements == me->q->capacity) {
        pthread_cond_wait(&me->var_prod, &me->m);
    }
    queue_add(me->q, value);
    pthread_mutex_unlock(&me->m);
    pthread_cond_signal(&me->var_cons);
}

static void
synced_queue_remove(synced_queue *me, void *value) {
    pthread_mutex_lock(&me->m);
    while (me->q->n_elements == 0) {
        pthread_cond_wait(&me->var_cons, &me->m);
    }
    queue_remove(me->q, value);
    pthread_mutex_unlock(&me->m);
    pthread_cond_signal(&me->var_prod);
}

// Two types of threads - one producer and one consumer. The producer
// sends jobs to the consumer.
typedef enum {
    PRODUCER = 0,
    CONSUMER = 1
} thread_type;

typedef struct {
    thread_type type;
    synced_queue *sq;
} prodcon_thread;

typedef struct {
    uint32_t x, y;
} job;


static int
msleep(long msec)
{
    struct timespec ts;
    int res;

    if (msec < 0)
    {
        errno = EINVAL;
        return -1;
    }
    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;
    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);
    return res;
}

static void
produce(synced_queue *sq) {
    for (size_t i = 0; i < 20; i++) {
        uint32_t x = rnd_pcg32_rand_range(50);
        uint32_t y = rnd_pcg32_rand_range(50);
        job j = (job){x, y};
        printf("produced job %u + %u\n", x, y);
        synced_queue_add(sq, &j);
        msleep(1000);
    }
    synced_queue_add(sq, &(job){0, 0});
}

static void
consume(synced_queue *sq) {
    while (true) {
        msleep(2000);
        job j;
        synced_queue_remove(sq, &j);
        if (j.x == 0 && j.y == 0) {
            printf("received finish\n");
            return;
        } else {
            printf("received job %u + %u = %u\n", j.x, j.y, j.x + j.y);
        }
    }
}

static void*
prodcon_fun(void *args) {
    prodcon_thread *me = (prodcon_thread *)args;
    if (me->type == PRODUCER) {
        produce(me->sq);
    } else {
        consume(me->sq);
    }
    return NULL;
}

int
main(int argc, char *argv[]) {
    rnd_pcg32_seed(1001, 370);
    synced_queue *sq = synced_queue_init(8, sizeof(job));
    thr_handle handles[2];
    prodcon_thread threads[2] = {
        {PRODUCER, sq}, {CONSUMER, sq}
    };
    assert(thr_create_threads(2, handles,
                              sizeof(prodcon_thread), &threads,
                              prodcon_fun));
    assert(thr_wait_for_threads(2, handles));
    synced_queue_free(sq);
    return 0;
}
