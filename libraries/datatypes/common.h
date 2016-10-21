#ifndef DATATYPES_COMMON_H
#define DATATYPES_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uintptr_t ptr;

#define NPTRS(n)  ((n) * sizeof(ptr))

void error(char *fmt, ...);

size_t rand_n(size_t n);

// It's shorter to type AT(foo) than *(ptr *)foo
#define AT(p) (*(ptr *)p)

// Basic logic
#define MAX(a, b) ((a > b) ? (a) : (b))
#define MIN(a, b) ((a > b) ? (b) : (a))

// Debug stuff
#define PRINT_RUN_INT(title, func) printf("=== %s\n", title); timed_run(&func); printf("\n")
#define PRINT_RUN(func) PRINT_RUN_INT(#func, func)

void timed_run(void (*func)());

// Bit munching
#define ALIGN(a, b) ((a + (b - 1)) & ~(b - 1))

#endif
