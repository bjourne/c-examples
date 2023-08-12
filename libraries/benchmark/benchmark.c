// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include "benchmark.h"
#include <stdio.h>

void
benchmark_print_settings() {
    printf("Compiler            : %s %d.%d.%d\n",
           BENCHMARK_CC, BENCHMARK_CC_VER);
    printf("C Target            : %s\n", BENCHMARK_TARGET);
    printf("CFLAGS              : %s\n", BENCHMARK_CFLAGS);
    printf("-march=native family: %s (actual: %s)\n",
           BENCHMARK_CC_NATIVE_FAMILY,
           BENCHMARK_CC_ACTUAL_FAMILY);
}
