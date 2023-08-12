// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#ifndef BENCHMARK_H
#define BENCHMARK_H

#if defined(__clang_major__)
#define BENCHMARK_CC_VER \
    __clang_major__, __clang_minor__, __clang_patchlevel__
#elif defined(__GNUC__)
#define BENCHMARK_CC_VER __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__
#else
#define BENCHMARK_CC_VER 0, 0, 0
#endif

#if defined(__x86_64__) && defined(__linux__)
#define BENCHMARK_TARGET "x86-64-linux"
#else
#define BENCHMARK_TARGET "unknown"
#endif

// Library containing utilities for writing benchmarks.
void benchmark_print_settings();




#endif
