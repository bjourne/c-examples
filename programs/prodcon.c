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
#include "threads/synced_queue.h"
#include "threads/threads.h"

// Two types of threads - one producer and one consumer. The producer
// sends jobs to the consumer.
typedef enum {
    PRODUCER = 0,
    CONSUMER = 1
} thread_type;

typedef struct {
    thr_handle handle;
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
    synced_queue *sq = synced_queue_init(8, sizeof(job), false);
    prodcon_thread threads[2] = {
        {0, PRODUCER, sq}, {0, CONSUMER, sq}
    };
    size_t tp_size = sizeof(prodcon_thread);
    assert(thr_create_threads2(2, tp_size, &threads, prodcon_fun));
    assert(thr_wait_for_threads2(2, tp_size, &threads));
    synced_queue_free(sq);
    return 0;
}
