// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates the producer-consumer pattern using my threads
// library.
#include <assert.h>
#include "threads/threads.h"

static void*
producer_thread(void *args) {
    return NULL;
}

int
main(int argc, char *argv[]) {
    thr_handle prod;
    assert(thr_create_threads(1, &prod, 0, NULL, producer_thread));

    return 0;
}
