#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdlib.h>

typedef uintptr_t ptr;

#define NPTRS(n)  ((n) * sizeof(ptr))

void error(char *fmt, ...);

#endif
