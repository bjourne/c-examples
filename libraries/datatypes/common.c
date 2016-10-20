#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void
error(char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    exit(1);
}

size_t
nano_count() {
  struct timespec t;
  int ret = clock_gettime(CLOCK_MONOTONIC, &t);
  if (ret != 0)
    error("clock_gettime failed", 0);
  return (size_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

void
timed_run(void (*func)()) {
    size_t start = nano_count();
    (func)();
    size_t end = nano_count();
    double secs = (double)(end - start) / 1000 / 1000 / 1000;
    printf("-> %.3f seconds\n", secs);
}

size_t
rand_n(size_t n) {
    return rand() % n;
}
