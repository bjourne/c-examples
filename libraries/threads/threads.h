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




#endif
