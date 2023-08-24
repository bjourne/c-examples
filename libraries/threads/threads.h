// Copyright (C) 2020, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef THREADS_H
#define THREADS_H

// A tiny abstraction layer on top of Windows threads and pthreads.

#include <stdbool.h>
#ifdef _WIN32
#include <windows.h>
typedef HANDLE thr_handle;
#else
#include <pthread.h>
typedef pthread_t thr_handle;
#endif


bool
thr_create_threads(size_t n, thr_handle *handles,
                   size_t size,
                   void *args,
                   void *(*func) (void *));
bool thr_wait_for_threads(size_t n, thr_handle *handles);

// This API is more convenient and perhaps better.
//
// size: size in bytes of each thread configuration.
// ctxs: points to an array of thread contexts, each of the given
// size. By convention, the first field in each context is a
// thr_handle.
bool
thr_create_threads2(size_t n, size_t size, void *ctxs, void *(*func) (void *));
bool
thr_wait_for_threads2(size_t n, size_t size, void *ctxs);


#endif
