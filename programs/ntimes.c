// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// This is my version of the "{n} times faster than C" benchmark
// presented in https://owen.cafe/posts/six-times-faster-than-c/
/*

  Some results:

  = M-5Y10 @ 0.8 GHz =

  == gcc 13.1.1, N_THREADS 4 ==

  count_naive          ->  5.74 seconds,  2.79 GB/s (count: 37600000)
  count_compl          ->  6.83 seconds,  2.34 GB/s (count: 37600000)
  count_blocked        ->  3.85 seconds,  4.16 GB/s (count: 37600000)
  count_lookup         -> 10.28 seconds,  1.56 GB/s (count: 37600000)
  count_avx2           ->  1.49 seconds, 10.71 GB/s (count: 37600000)
  count_avx2_2         ->  1.49 seconds, 10.75 GB/s (count: 37600000)
  count_thr_avx2       ->  2.18 seconds,  7.33 GB/s (count: 37600000)
  count_thr_blocked    ->  3.13 seconds,  5.11 GB/s (count: 37600000)
  count_thr_naive      ->  4.33 seconds,  3.70 GB/s (count: 37600000)

  == clang 15.0.7, N_THREADS 4 ==

  count_naive          ->  4.40 seconds,  3.64 GB/s (count: 37600000)
  count_compl          ->  6.32 seconds,  2.53 GB/s (count: 37600000)
  count_blocked        ->  4.47 seconds,  3.58 GB/s (count: 37600000)
  count_lookup         -> 13.05 seconds,  1.23 GB/s (count: 37600000)
  count_avx2           ->  1.85 seconds,  8.64 GB/s (count: 37600000)
  count_avx2_2         ->  1.92 seconds,  8.35 GB/s (count: 37600000)
  count_thr_avx2       ->  2.16 seconds,  7.39 GB/s (count: 37600000)
  count_thr_blocked    ->  3.47 seconds,  4.62 GB/s (count: 37600000)
  count_thr_naive      ->  3.61 seconds,  4.43 GB/s (count: 37600000)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "random/random.h"

int count_avx2(const char *s);
int count_avx2_2(const char *s);
int count_blocked(const char *s);
int count_compl(const char* input);
int count_lookup(const char *input);
int count_naive(const char* input);
int count_switches(const char *input);

// Threaded versions
int count_thr_avx2(const char *s);
int count_thr_blocked(const char *);
int count_thr_naive(const char *s);

#define N_RND_BUF 100000

#define N_BUF 3L * 1000L * 1000 * 1000
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
    printf("%-20s -> %5.2f seconds, %5.2f GB/s (count: %ld)\n",
           name, secs, tot_gbs / secs, cnt);
}

int
main(int argc, char *argv[]) {
    rnd_pcg32_seed(1001, 370);
    uint32_t *rnd_buf = malloc_aligned(64, sizeof(uint32_t) * N_RND_BUF);
    rnd_pcg32_rand_range_fill(rnd_buf, 10, N_RND_BUF);

    printf("Allocating %6.2fGB\n", N_GIG);
    char *buf = malloc_aligned(64, N_BUF);

    printf("Filling buffer\n");
    setbuf(stdout, NULL);
    for (size_t i = 0; i < N_BUF - 1; i++) {
        int v = rnd_buf[i % N_RND_BUF];
        if (v < 2) {
            buf[i] = 'p';
        } else if (v >= 2 && v < 4) {
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
    benchmark("count_avx2", count_avx2, buf);
    benchmark("count_avx2_2", count_avx2_2, buf);
    benchmark("count_thr_avx2", count_thr_avx2, buf);
    benchmark("count_thr_blocked", count_thr_blocked, buf);
    benchmark("count_thr_naive", count_thr_naive, buf);

    free(rnd_buf);
    free(buf);

    return 0;
}
