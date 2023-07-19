// Copyright (C) 2020, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <inttypes.h>
#include <stdio.h>
#include "threads/threads.h"

bool
thr_create_threads(size_t n, thr_handle *handles,
                   size_t size, void *args,
                   void *(*func) (void *)) {
    for (size_t i = 0; i < n; i++) {
        void *arg = (void *)((char*)args + size * i);
#if _WIN32
        handles[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func,
                                  arg, 0, NULL);
        if (!handles[i]) {
            return false;
        }
#else
        if (pthread_create(&handles[i], NULL, func, arg)) {
            return false;
        }
#endif
    }
    return true;
}

bool
thr_wait_for_threads(size_t n, thr_handle *handles) {
    for (size_t i = 0; i < n; i++) {
#if _WIN32
        WaitForSingleObject(handles[i], INFINITE);
#else
        if (pthread_join(handles[i], NULL)) {
            return false;
        }
#endif
    }
    return true;
}
