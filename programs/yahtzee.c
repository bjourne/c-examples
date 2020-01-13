// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

//////////////////////////////////////////////////////////////////////////////
// Range of numbers, number of numbers and number of parser threads.
//////////////////////////////////////////////////////////////////////////////
#define MIN_VALUE (1 * 1000 * 1000)
#define MAX_VALUE (2 * 1000 * 1000)
#define N_VALUES 400 * 1000 * 1000
#define N_THREADS 8

//////////////////////////////////////////////////////////////////////////////
// Timing functions.
//////////////////////////////////////////////////////////////////////////////
#define NS_TO_S(x)      ((double)(x) / 1000 / 1000 / 1000)

uint64_t
nano_count() {
#ifdef _WIN32
    static double scale_factor;

    static uint64_t hi = 0;
    static uint64_t lo = 0;

    LARGE_INTEGER count;
    BOOL ret = QueryPerformanceCounter(&count);
    if (ret == 0) {
        printf("QueryPerformanceCounter failed.\n");
        abort();
    }

    if (scale_factor == 0.0) {
        LARGE_INTEGER frequency;
        BOOL ret = QueryPerformanceFrequency(&frequency);
        if (ret == 0) {
            printf("QueryPerformanceFrequency failed.\n");
            abort();
        }
        scale_factor = (1000000000.0 / frequency.QuadPart);
  }
#ifdef CPU_64
    hi = count.HighPart;
#else
    if (lo > count.LowPart) {
        hi++;
    }
#endif
    lo = count.LowPart;
    return (uint64_t)(((hi << 32) | lo) * scale_factor);
#else
    struct timespec t;
    int ret = clock_gettime(CLOCK_MONOTONIC, &t);
    if (ret != 0) {
        printf("clock_gettime failed.\n");
        abort();
    }
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
#endif
}

//////////////////////////////////////////////////////////////////////////////
// Generate the data file.
//////////////////////////////////////////////////////////////////////////////
static int
rand_in_range(int lo, int hi) {
    int range = hi - lo;
    int val = (rand() & 0xff) << 16 |
        (rand() & 0xff) << 8 |
        (rand() & 0xff);
    return (val % range) + lo;
}

static void
run_generate(const char *path) {
    srand(1234);
    FILE *f = fopen(path, "wb");
    for (int i = 0; i < N_VALUES; i++) {
        fprintf(f, "%d\n", rand_in_range(MIN_VALUE, MAX_VALUE));
    }
    fclose(f);
}

//////////////////////////////////////////////////////////////////////////////
// Fast number parser using loop unrolling macros.
//////////////////////////////////////////////////////////////////////////////
#define PARSE_FIRST_DIGIT               \
    if (*buf >= '0') {                  \
        val = *buf++ - '0';             \
    } else {                            \
        goto done;                      \
    }
#define PARSE_NEXT_DIGIT                \
    if (*buf >= '0') {                  \
        val = val*10 + *buf++ - '0';    \
    } else {                            \
        goto done;                      \
    }

static void
parse_chunk(char *buf, const char *end, size_t *accum) {
    size_t val = 0;
    while (buf < end) {
        // Parse up to 7 digits.
        PARSE_FIRST_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
    done:
        #ifdef _WIN32
        InterlockedExchangeAdd64(&accum[val], val);
        #else
        __sync_fetch_and_add(&accum[val], val);
        #endif
        // Skip newline character.
        buf++;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Thread definition
//////////////////////////////////////////////////////////////////////////////
typedef struct {
    char *chunk_start;
    char *chunk_end;
    size_t *accum;
} parse_chunk_thread_args;

#ifdef _WIN32
static DWORD WINAPI
parse_chunk_thread(LPVOID args) {
    parse_chunk_thread_args *a = (parse_chunk_thread_args *)args;
    parse_chunk(a->chunk_start, a->chunk_end, a->accum);
    return 0;
}
#else
static void*
parse_chunk_thread(void *args) {
    parse_chunk_thread_args *a = (parse_chunk_thread_args *)args;
    parse_chunk(a->chunk_start, a->chunk_end, a->accum);
    return NULL;
}
#endif

//////////////////////////////////////////////////////////////////////////////
// Parse the whole file.
//////////////////////////////////////////////////////////////////////////////
static void
run_test(const char *path) {
    uint64_t time_start = nano_count();

    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long int n_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf_start = (char *)malloc(sizeof(char) * n_bytes);
    char *buf_end = buf_start + n_bytes;
    assert(fread(buf_start, 1, n_bytes, f) == n_bytes);
    fclose(f);

    uint64_t time_read = nano_count();

    char *chunks[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        chunks[i] = buf_start + (n_bytes / N_THREADS) * i;
        if (i > 0) {
            // Adjust the chunks starting points until they reach past
            // a newline.
            while (*chunks[i] != '\n') {
                chunks[i]++;
            }
            chunks[i]++;
        }
    }
    size_t *accum = calloc(MAX_VALUE, sizeof(size_t));

    #if _WIN32
    HANDLE threads[N_THREADS];
    #else
    pthread_t threads[N_THREADS];
    #endif
    parse_chunk_thread_args args[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        char *chunk_start = chunks[i];
        char *chunk_end = buf_end;
        if (i < N_THREADS - 1) {
            chunk_end = chunks[i + 1];
        }
        args[i].chunk_start = chunk_start;
        args[i].chunk_end = chunk_end;
        args[i].accum = accum;
        #if _WIN32
        threads[i] = CreateThread(NULL, 0, parse_chunk_thread,
                                  &args[i], 0, NULL);
        #else
        pthread_create(&threads[i], NULL, parse_chunk_thread, &args[i]);
        #endif
    }
    for (int i = 0; i < N_THREADS; i++) {
        #if _WIN32
        WaitForSingleObject(threads[i], INFINITE);
        #else
        pthread_join(threads[i], NULL);
        #endif
    }

    uint64_t time_parsed = nano_count();
    size_t max = 0;
    for (int i = 0; i < MAX_VALUE; i++) {
        size_t val = accum[i];
        if (val > max) {
            max = val;
        }
    }

    uint64_t time_accum = nano_count();

    printf("max = %zu\n", max);
    free(accum);
    free(buf_start);

    // Print timings.
    double read_secs = NS_TO_S(time_read - time_start);
    double parse_secs = NS_TO_S(time_parsed - time_read);
    double accum_secs = NS_TO_S(time_accum - time_parsed);
    double total_secs = NS_TO_S(time_accum - time_start);
    printf("Read : %.3f seconds\n", read_secs);
    printf("Parse: %.3f seconds\n", parse_secs);
    printf("Accum: %.3f seconds\n", accum_secs);
    printf("Total: %.3f seconds\n", total_secs);
}

int
main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("%s: [generate|test] path\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *cmd = argv[1];
    if (strcmp(cmd, "generate") == 0) {
        run_generate(argv[2]);
    } else if (strcmp(cmd, "test") == 0) {
        run_test(argv[2]);
    } else {
        printf("%s: [generate|test] path\n", argv[0]);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
