#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uintptr_t ptr;

#define NPTRS(n)  ((n) * sizeof(ptr))

void error(char *fmt, ...);

// Debug stuff
#define PRINT_RUN(func) printf("=== %s\n", #func); timed_run(&func)
void timed_run(void (*func)());

// Bit munching
#define ALIGN(a, b) ((a + (b - 1)) & ~(b - 1))

#endif
