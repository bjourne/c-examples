// Copyright (C) 2019-2020, 2022-2023 Björn A. Lindqvist
#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

void *
malloc_aligned(size_t alignment, size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, size)) {
        return NULL;
    }
    return p;
}

void
error(char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    abort();
}

uint64_t
nano_count() {
#ifdef _WIN32
    static double scale_factor;

    static uint64_t hi = 0;
    static uint64_t lo = 0;

    LARGE_INTEGER count;
    BOOL ret = QueryPerformanceCounter(&count);
    if (ret == 0) {
        error("QueryPerformanceCounter failed");
    }

    if (scale_factor == 0.0) {
        LARGE_INTEGER frequency;
        BOOL ret = QueryPerformanceFrequency(&frequency);
        if (ret == 0) {
            error("QueryPerformanceFrequency failed");
        }
        scale_factor = (1000000000.0 / frequency.QuadPart);
  }
#ifdef CPU_64
    hi = count.HighPart;
#else
    if (lo > count.LowPart) {
        hi++;
    }
#endif
    lo = count.LowPart;
    return (uint64_t)(((hi << 32) | lo) * scale_factor);
#else
    struct timespec t;
    int ret = clock_gettime(CLOCK_MONOTONIC, &t);
    if (ret != 0) {
        error("clock_gettime failed");
    }
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
#endif
}

double
nanos_to_secs(uint64_t nanos) {
    return (double)nanos / 1000 / 1000 / 1000;
}

void
timed_run(void (*func)()) {
    uint64_t start = nano_count();
    (func)();
    uint64_t end = nano_count();
    double secs = nanos_to_secs(end - start);
    printf("-> %.3f seconds\n", secs);
}

int
rand_n(int n) {
    return rand() % n;
}

void
rand_init(unsigned int seed) {
    if (!seed) {
        seed = (unsigned int)time(NULL);
    }
    srand(seed);
}

void
sleep_cp(unsigned int millis) {
    #ifdef _WIN32
        Sleep(millis);
    #else
        usleep(millis * 1000);
    #endif // _WIN32
}
