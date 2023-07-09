// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// This is my version of the "{n} times faster than C" benchmark
// presented in https://owen.cafe/posts/six-times-faster-than-c/
/*

  Some results:

  = M-5Y10 @ 0.8 GHz =

  == gcc 13.1.1 ==

  count_naive     ->  1.45 seconds,  2.76 GB/s (count: 102584)
  count_compl     ->  1.72 seconds,  2.33 GB/s (count: 102584)
  count_blocked   ->  0.98 seconds,  4.06 GB/s (count: 102584) [BITS = 15]
  count_lookup    ->  2.59 seconds,  1.55 GB/s (count: 102584)
  count_avx2      ->  0.40 seconds,  9.93 GB/s (count: 102584)

  == clang 15.0.7 ==

  count_naive     ->  1.12 seconds,  3.58 GB/s (count: 102584)
  count_compl     ->  1.58 seconds,  2.53 GB/s (count: 102584)
  count_blocked   ->  1.14 seconds,  3.52 GB/s (count: 102584) [BITS = 15]
  count_lookup    ->  3.27 seconds,  1.22 GB/s (count: 102584)
  count_avx2      ->  0.48 seconds,  8.28 GB/s (count: 102584)

 */


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "random/random.h"

int count_compl(const char* input);
int count_naive(const char* input);
int count_switches(const char *input);
int count_lookup(const char *input);
int count_blocked(const char *s);
int count_avx2(const char *s);
int count_avx2_2(const char *s);

#define N_BUF (500L * 1000 * 1000)
#define N_GIG ((double)N_BUF / (1000 * 1000 * 1000))
#define N_REPS 8L

void
benchmark(const char *name, int (*func)(const char *s), const char *s) {
    size_t start = nano_count();
    size_t cnt = 0;
    for (int i = 0; i < N_REPS; i++) {
        cnt += func(s);
    }
    size_t end = nano_count();
    double secs = (double)(end - start) / (1000 * 1000 * 1000);
    double tot_gbs = N_GIG * N_REPS;
    printf("%-15s -> %5.2f seconds, %5.2f GB/s (count: %ld)\n",
           name, secs, tot_gbs / secs, cnt);
}

int
main(int argc, char *argv[]) {
    rnd_pcg32_seed(1007, 370);
    printf("Allocating %6.2fGB\n", N_GIG);
    char *buf = malloc_aligned(64, N_BUF);

    printf("Filling buffer\n");
    setbuf(stdout, NULL);
    for (size_t i = 0; i < N_BUF - 1; i++) {
        int v = rnd_pcg32_rand_range(10);
        if (v == 0) {
            buf[i] = 'p';
        } else if (v == 1) {
            buf[i] = 's';
        } else {
            buf[i] = 'a';
        }
        if (i % (N_BUF / 100) == 0) {
            printf("%ld%%... ", i / (N_BUF / 100));
        }
    }
    printf(" 100%%\n");
    buf[N_BUF - 1] = '\0';
    printf("Setup done, benchmarking...\n");


    benchmark("count_naive", count_naive, buf);
    benchmark("count_compl", count_compl, buf);
    benchmark("count_blocked", count_blocked, buf);
    benchmark("count_lookup", count_lookup, buf);
    benchmark("count_avx2_2", count_avx2_2, buf);
    benchmark("count_avx2", count_avx2, buf);

    free(buf);
    return 0;
}
