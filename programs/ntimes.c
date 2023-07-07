// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// This is my version of the "{n} times faster than C" benchmark
// presented in https://owen.cafe/posts/six-times-faster-than-c/
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "random/random.h"

int
count_compl(const char* input) {
    int res = 0;
    while ((uintptr_t) input % sizeof(size_t)) {
        char c = *input++;
        res += c == 's';
        res -= c == 'p';
        if (c == 0) return res;
    }

    const size_t ONES = ((size_t) -1) / 255;  // 0x...01010101
    const size_t HIGH_BITS = ONES << 7;       // 0x...80808080
    const size_t SMASK = ONES * (size_t) 's'; // 0x...73737373
    const size_t PMASK = ONES * (size_t) 'p'; // 0x...70707070
    size_t s_accum = 0;
    size_t p_accum = 0;
    int iters = 0;
    while (1) {
        size_t w;
        memcpy(&w, input, sizeof(size_t));
        if ((w - ONES) & ~w & HIGH_BITS) break;
        input += sizeof(size_t);

        size_t s_high_bits = ((w ^ SMASK) - ONES) & ~(w ^ SMASK) & HIGH_BITS;
        size_t p_high_bits = ((w ^ PMASK) - ONES) & ~(w ^ PMASK) & HIGH_BITS;

        s_accum += s_high_bits >> 7;
        p_accum += p_high_bits >> 7;
        if (++iters >= 255 / sizeof(size_t)) {
            res += s_accum % 255;
            res -= p_accum % 255;
            iters = s_accum = p_accum = 0;
        }
    }
    res += s_accum % 255;
    res -= p_accum % 255;

    while (1) {
        char c = *input++;
        res += c == 's';
        res -= c == 'p';
        if (c == 0)
            break;
    }
    return res;
}

int
count_naive(const char* input) {
    size_t len = strlen(input);
    int res = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = input[i];
        res += c == 's';
        res -= c == 'p';
    }
    return res;
}

int
count_switches(const char *input) {
    int res = 0;
    while (true) {
        char c = *input++;
        switch (c) {
        case '\0':
            return res;
        case 's':
            res += 1;
            break;
        case 'p':
            res -= 1;
            break;
        default:
            break;
        }
    }
}

static
int to_add[256] = {
  ['s'] = 1,
  ['p'] = -1,
};

int
count_lookup(const char *input) {
    int res = 0;
    while (true) {
        char c = *input++;
        if (c == '\0') {
            return res;
        } else {
            res += to_add[(int) c];
        }
    }
}

#define N_BLOCK_BITS 6
#define N_BLOCK (1 << N_BLOCK_BITS)
#define N_BLOCK_MASK (N_BLOCK - 1)

int
count_blocked(const char *s) {
    int r = 0;
    int tmp = 0;
    size_t n = strlen(s);
    for (size_t i = n & N_BLOCK_MASK; i--; ++s) {
        tmp += (*s == 's') - (*s == 'p');
    }
    r += tmp;

    for (n >>= N_BLOCK_BITS; n--;) {
        tmp = 0;
        for (int i = N_BLOCK; i--; ++s) {
            tmp += (*s == 's') - (*s == 'p');
        }
        r += tmp;
    }
    return r;
}

#define N_BUF (1L * 1000 * 1000 * 1000)
#define N_GIG ((double)N_BUF / (1000 * 1000 * 1000))
#define N_REPS 1L

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

    printf("%-15s -> %4.2f seconds, %5.2f GB/s (count: %ld)\n",
           name, secs, tot_gbs / secs, cnt);
}

int
main(int argc, char *argv[]) {
    rnd_pcg32_seed(1007, 37);
    printf("Allocating %6.2fGB\n", N_GIG);
    char *buf = malloc_aligned(64, N_BUF);

    printf("Filling buffer\n");
    setbuf(stdout, NULL);
    for (size_t i = 0; i < N_BUF - 1; i++) {
        int v = rnd_pcg32_rand_range(2);
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
    printf("\n");
    buf[N_BUF - 1] = '\0';
    printf("Setup done, benchmarking...\n");


    benchmark("count_naive", count_naive, buf);
    benchmark("count_switches", count_switches, buf);
    benchmark("count_compl", count_compl, buf);
    benchmark("count_blocked", count_blocked, buf);
    benchmark("count_lookup", count_lookup, buf);

    free(buf);
    return 0;
}
