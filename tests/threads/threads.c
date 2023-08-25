// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include "datatypes/common.h"
#include "threads/threads.h"
#include "threads/synced_queue.h"

#define N_JOBS (1 * 1000 * 1000)

typedef struct {
    uint32_t x, y;
} job;

typedef struct {
    thr_handle handle;
    bool is_producer;
    synced_queue *sq;
    uint64_t sum;
} thread_ctx;

static void*
fun(void *args) {
    thread_ctx *ctx = (thread_ctx *)args;
    if (ctx->is_producer) {
        for (uint32_t i = 0; i < N_JOBS; i++) {
            uint32_t x = 1 + i;
            uint32_t y = 1 + i;
            synced_queue_add(ctx->sq, &(job){x, y});
        }
        synced_queue_add(ctx->sq, &(job){0, 0});
    } else {
        while (true) {
            job j;
            synced_queue_remove(ctx->sq, &j);
            ctx->sum += j.x + j.y;
            if (j.x == 0 && j.y == 0) {
                break;
            }
        }
    }
    return NULL;
}

static uint64_t
run_case(synced_queue_lock_type lock_type, size_t queue_size) {
    uint64_t start = nano_count();
    synced_queue *sq = synced_queue_init(queue_size, sizeof(job), lock_type);
    thread_ctx ctx[2] = {
        {0, true, sq, 0}, {0, false, sq, 0}
    };
    assert(thr_create_threads2(2, sizeof(thread_ctx), &ctx, fun));
    assert(thr_wait_for_threads2(2, sizeof(thread_ctx), &ctx));
    synced_queue_free(sq);
    return nano_count() - start;
}

// Some numbers on my machine:
//
//      spinlock                256 elements  0.15 us/job
//      uncontested spinlock    256 elements  0.14 us/job
//      mutex                   256 elements  0.28 us/job
void
benchmark_synced_queue() {

    size_t queue_sizes[] = {1024, 512, 256, 128, 64, 32, 16, 8, 4};
    synced_queue_lock_type lock_types[] = {
        SYNCED_QUEUE_SPIN_LOCK,
        SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED,
        SYNCED_QUEUE_MUTEX
    };
    for (size_t j = 0; j < ARRAY_SIZE(queue_sizes); j++) {
        for (size_t i = 0; i < ARRAY_SIZE(lock_types); i++) {
            synced_queue_lock_type lt = lock_types[i];
            size_t queue_size = queue_sizes[j];
            uint64_t nanos = run_case(lt, queue_size);
            double us_per_job = ((double)nanos / 1000.0) / (double)N_JOBS;
            char *lt_name = "mutex";
            if (lt == SYNCED_QUEUE_SPIN_LOCK_UNCONTESTED) {
                lt_name = "uncontested spinlock";
            } else if (lt == SYNCED_QUEUE_SPIN_LOCK) {
                lt_name = "spinlock";
            }
            printf("%-22s %4ld elements %5.2lf us/job\n",
                   lt_name, queue_size, us_per_job);
        }
    }
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(benchmark_synced_queue);
    return 0;
}
