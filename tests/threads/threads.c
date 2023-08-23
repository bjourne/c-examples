// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include "datatypes/common.h"
#include "threads/threads.h"
#include "threads/synced_queue.h"

#define N_JOBS (10 * 1000 * 1000)

typedef struct {
    uint32_t x, y;
} job;

typedef struct {
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

void
benchmark_synced_queue() {
    synced_queue *sq = synced_queue_init(128, sizeof(job), true);
    thr_handle handles[2];
    thread_ctx ctx[2] = {
        {true, sq, 0}, {false, sq, 0}
    };
    assert(thr_create_threads(2, handles, sizeof(thread_ctx), &ctx, fun));
    assert(thr_wait_for_threads(2, handles));
    synced_queue_free(sq);
    printf("%ld\n", ctx[1].sum);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(benchmark_synced_queue);
    return 0;
}
