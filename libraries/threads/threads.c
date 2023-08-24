// Copyright (C) 2020, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <inttypes.h>
#include <stdio.h>
#include "threads/threads.h"

static bool
create(thr_handle *handle, void *ctx, void *(*func)(void *)) {
#if _WIN32
    *handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func,
                           ctx, 0, NULL);
    if (!*handle) {
        return false;
    }
#else
    if (pthread_create(handle, NULL, func, ctx)) {
        return false;
    }
#endif
    return true;
}

static bool
wait(thr_handle handle) {
#if _WIN32
    WaitForSingleObject(handles, INFINITE);
#else
    if (pthread_join(handle, NULL)) {
        return false;
    }
#endif
    return true;
}

bool
thr_create_threads2(size_t n, size_t size, void *ctxs, void *(*func) (void *)) {
    for (size_t i = 0; i < n; i++) {
        void *arg = (void *)((char*)ctxs + size * i);
        if (!create(arg, arg, func)) {
            return false;
        }
    }
    return true;
}

bool
thr_wait_for_threads2(size_t n, size_t size, void *ctxs) {
    for (size_t i = 0; i < n; i++) {
        thr_handle *handle = (thr_handle *)((char*)ctxs + size * i);
        if (!wait(*handle)) {
            return false;
        }
    }
    return true;
}

bool
thr_create_threads(size_t n, thr_handle *handles,
                   size_t size, void *args,
                   void *(*func) (void *)) {
    for (size_t i = 0; i < n; i++) {
        void *arg = (void *)((char*)args + size * i);
        if (!create(&handles[i], arg, func)) {
            return false;
        }
    }
    return true;
}

bool
thr_wait_for_threads(size_t n, thr_handle *handles) {
    for (size_t i = 0; i < n; i++) {
        if (!wait(handles[i])) {
            return false;
        }
    }
    return true;
}
