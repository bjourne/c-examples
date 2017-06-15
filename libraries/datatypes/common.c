#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

void
error(char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    abort();
}

size_t
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
    return (size_t)t.tv_sec * 1000000000 + t.tv_nsec;
#endif
}

void
timed_run(void (*func)()) {
    size_t start = nano_count();
    (func)();
    size_t end = nano_count();
    double secs = (double)(end - start) / 1000 / 1000 / 1000;
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
rand_shuffle(void *array, size_t n, size_t size) {
    char tmp[size];
    char *arr = array;
    size_t stride = size * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}
