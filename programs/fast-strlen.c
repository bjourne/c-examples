// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Compile and run with:
//
//   gcc -O3 -march=native fast-strlen.c -lpthread -o fast-strlen
//       && ./fast-strlen
//
// Use gcc because clang is too smart and optimizes away parts of the
// benchmark. Results on Xeon(R) CPU E5-2650 v4 @ 2.20GHz with gcc
// 9.4.0:
//
//   Scanning 10 times over 4.00GB...
//   strlen_avx2      ->  5.09 seconds,  7.86 GB/s (length: 4000007776)
//   threaded_strlen  ->  1.17 seconds, 34.09 GB/s (length: 4000007776)
//   slow_strlen      -> 18.34 seconds,  2.18 GB/s (length: 4000007776)
//   strlen           ->  4.56 seconds,  8.76 GB/s (length: 4000007776)
//
// How? The function threaded_strlen intentionally triggers a page
// fault to find an upper bound for the length of the string. It
// then has a set of threads run memchr in parallel on non-overlapping
// chunks of memory.
//
// Why? For fun. I was nerd sniped by the {n} times benchmark [1] and
// found that performance was limited by strlen (the best variant [2]
// is within 10% of strlen on the buffer). So if you make strlen
// faster that benchmark can be faster too. Besides, I haven't seen
// any multithreaded strlen's before.
//
// Obviously, this technique heavily relies on undefined behavior and
// shouldn't be used for anything other than beating benchmarks.
//
// [1]: https://owen.cafe/posts/six-times-faster-than-c/
// [2]: https://gist.github.com/Const-me/3ade77faad47f0fbb0538965ae7f8e04
#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Size of buffer to test with and number of times the benchmarked
// function is called.
#define N_BUF (2L * 1000 * 1000 * 1000)
#define N_REPS 10
#define N_GIG ((double)N_BUF / (1000 * 1000 * 1000))

__attribute__((noinline)) size_t
slow_strlen(const char *s) {
    size_t i = 0;
    while (s[i++] != '\0');
    return --i;
}

#ifdef __AVX2__
__attribute__((noinline)) size_t
strlen_avx2(const char *s) {
    assert((uintptr_t)s % 32 == 0);
    const __m256i zero = _mm256_setzero_si256();
    const char *p = s;
    while (true) {
        __m256i mask = _mm256_cmpeq_epi8(*(const __m256i *)p, zero);
        if (!_mm256_testz_si256(mask, mask)) {
            return p - s + slow_strlen(p);
        }
        p += 32;
    }
}
#endif

#ifdef __AVX512F__
__attribute__((noinline)) size_t
strlen_avx512(const char *s) {
    assert((uintptr_t)s % 64 == 0);
    const __m512i zero = _mm512_setzero_epi32();
    const char *p = s;
    while (true) {
        __m512i v = _mm512_loadu_si512(p);
        __mmask64 mask = _mm512_cmpeq_epi8_mask(v, zero);
        if (mask) {
            return p - s + slow_strlen(p);
        }
        p += 64;
    }
}
#endif

typedef struct {
    const char *s;
    size_t n_bytes;
    size_t cnt;
    bool found;
} strlen_job;

// Limit is 256TB
#define SIGSEGV_SWEEP_SPACE 256L * 1000 * 1000 * 1000 * 1000
#define SIGSEGV_SWEEP_STEP 4L * 1000 * 1000;

static char *
sigsegv_addr = NULL;

static void *
thr_sigsegv(void *arg) {
    size_t at = SIGSEGV_SWEEP_STEP;
    const char *s = (const char *)arg;
    size_t cnt = 0;

    // Without the conditional the compiler may assume the loop is
    // infinite and optimize away the memory read.
    while (at < SIGSEGV_SWEEP_SPACE) {
        if (!s[at]) {
            cnt++;
        }
        at += SIGSEGV_SWEEP_STEP;
    }
    return (void *)cnt;
}

// Plain-old memchr appears to work the best when the length is
// limited.
static void *
thr_strlen(void *arg) {
    strlen_job *job = (strlen_job *)arg;
    const char *s = job->s;
    const char *p = memchr(s, '\0', job->n_bytes);
    job->found = p != NULL;
    job->cnt = p - s;
    return NULL;
}

static void
sigsegv_record(int signal, siginfo_t* siginfo, void* uap) {
    sigsegv_addr = siginfo->si_addr;
    pthread_exit(NULL);
}

static size_t
ceil_div(size_t a, size_t b) {
    return a / b + (a % b != 0);
}

__attribute__((noinline)) size_t
threaded_strlen(const char *s) {
    struct sigaction act, oldact;
    memset(&act, 0, sizeof(struct sigaction));
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    act.sa_sigaction = sigsegv_record;

    // Launch a thread whose only job is to page fault to give us an
    // upper bound on the number of bytes to search.
    pthread_t h;
    sigaction(SIGSEGV, &act, &oldact);
    assert(!pthread_create(&h, NULL, thr_sigsegv, (void *)s));
    assert(!pthread_join(h, NULL));
    size_t n_bytes = sigsegv_addr - s;

    // Launch a thread pool scanning for '\0' in disjoint regions of
    // memory in parallel.
    int n_threads = sysconf(_SC_NPROCESSORS_ONLN);
    size_t n_bytes_per_thread = ceil_div(n_bytes, n_threads);

    strlen_job *jobs = malloc(sizeof(strlen_job) * n_threads);
    pthread_t *handles = malloc(sizeof(pthread_t) * n_threads);
    for (int i = 0; i < n_threads; i++) {
        jobs[i] = (strlen_job){
            s + i * n_bytes_per_thread,
            n_bytes_per_thread,
            0,
            false
        };
        assert(!pthread_create(&handles[i],
                               NULL,
                               thr_strlen,
                               (void *)&jobs[i]));
    }
    for (int i = 0; i < n_threads; i++) {
        assert(!pthread_join(handles[i], NULL));
    }
    // Restore old handler here -- thr_strlen can also fault if
    // the buffer is small.
    sigaction(SIGSEGV, &oldact, NULL);

    for (int i = 0; i < n_threads; i++) {
        strlen_job j = jobs[i];
        if (j.found) {
            free(jobs);
            free(handles);
            return i * n_bytes_per_thread + j.cnt;
        }
    }
    assert(false);
}

static uint64_t
nano_count() {
    struct timespec t;
    assert(!clock_gettime(CLOCK_MONOTONIC, &t));
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

void
benchmark(const char *name,
          size_t (*func)(const char *s),
          const char *s) {
    size_t start = nano_count();
    size_t cnt = func(s);
    for (int i = 1; i < N_REPS; i++) {
        assert(func(s) == cnt);
    }
    size_t end = nano_count();
    double secs = (double)(end - start) / (1000 * 1000 * 1000);
    double tot_gbs = N_GIG * N_REPS;
    printf("%-15s -> %5.2f seconds, %5.2f GB/s (length: %ld)\n",
           name, secs, tot_gbs / secs, cnt);
}

int
main(int argc, char *argv[]) {
    char *buf = NULL;
    assert(!posix_memalign((void *)&buf, 64, N_BUF));
    memset(buf, 'A', N_BUF);
    buf[N_BUF - 1] = '\0';

    printf("Scanning %d times over %.2fGB...\n", N_REPS, N_GIG);
    #ifdef __AVX2__
    benchmark("strlen_avx2", strlen_avx2, buf);
    #endif
    #ifdef __AVX512F__
    benchmark("strlen_avx512", strlen_avx512, buf);
    #endif
    benchmark("threaded_strlen", threaded_strlen, buf);
    benchmark("slow_strlen", slow_strlen, buf);
    benchmark("strlen", strlen, buf);
    free(buf);
    return 0;
}
