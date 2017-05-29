#ifndef DATATYPES_COMMON_H
#define DATATYPES_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uintptr_t ptr;

#define NPTRS(n)  ((n) * sizeof(ptr))

void error(char *fmt, ...);

int rand_n(int n);
void rand_init(unsigned int seed);

// Platform detection
#if defined(__amd64__) || defined(__x86_64__) || defined(_M_AMD64)
#define CPU_64
#else
#define CPU_32
#endif

// It's shorter to type AT(foo) than *(ptr *)foo
#define AT(p) (*(ptr *)(p))

// Basic logic
#define MAX(a, b) ((a > b) ? (a) : (b))
#define MIN(a, b) ((a > b) ? (b) : (a))

#define CLAMP(a, lo, hi) MIN(MAX(a, lo), hi)

// Utility
#define ARRAY_SIZE(a)       (sizeof((a))/sizeof((a)[0]))

// Debug stuff
#define PRINT_RUN_INT(title, func) printf("=== %s\n", title); timed_run(&func); printf("\n")
#define PRINT_RUN(func) PRINT_RUN_INT(#func, func)

void timed_run(void (*func)());

// Timing
size_t nano_count();

#endif
