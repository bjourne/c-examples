// Copyright (C) 2020, 2023 Björn Lindqvist <bjourne@gmail.com>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include "datatypes/common.h"
#include "threads/threads.h"

//////////////////////////////////////////////////////////////////////////////
// Range of numbers, number of numbers and number of parser threads.
//////////////////////////////////////////////////////////////////////////////
#define MIN_VALUE (1 * 1000 * 1000)
#define MAX_VALUE (2 * 1000 * 1000)
#define N_VALUES 100 * 1000 * 1000
#define N_THREADS 4

//////////////////////////////////////////////////////////////////////////////
// Timing functions.
//////////////////////////////////////////////////////////////////////////////
#define NS_TO_S(x)      ((double)(x) / 1000 / 1000 / 1000)

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
#define PARSE_FIRST_DIGIT              \
    if (*at >= '0')         {          \
        val = *at++ - '0';             \
    } else {                           \
        goto done;                     \
    }
#define PARSE_NEXT_DIGIT               \
    if (*at >= '0') {                  \
        val = val*10 + *at++ - '0';    \
    } else {                           \
        goto done;                     \
    }

static void
parse_chunk(char *at, const char *end, int *accum) {
    int val = 0;
    while (at < end) {
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
        InterlockedExchangeAdd(&accum[val], 1);
        #else
        __sync_fetch_and_add(&accum[val], 1);
        #endif
        // Skip newline character.
        at++;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Thread definition
//////////////////////////////////////////////////////////////////////////////
typedef struct {
    char *chunk_start;
    char *chunk_end;
    int *accum;
} parse_chunk_thread_args;

static void*
parse_chunk_thread(void *args) {
    parse_chunk_thread_args *a = (parse_chunk_thread_args *)args;
    parse_chunk(a->chunk_start, a->chunk_end, a->accum);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////
// Parse the whole file.
//////////////////////////////////////////////////////////////////////////////
static bool
run_test(const char *path) {
    uint64_t time_start = nano_count();

#ifdef _WIN32
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    uint64_t n_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf_start = (char *)malloc(sizeof(char) * n_bytes);
    if (fread(buf_start, 1, n_bytes, f) != n_bytes) {
        free(buf_start);
        return false;
    }
    fclose(f);
#else
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        return false;
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        return false;
    }
    uint64_t n_bytes = sb.st_size;
    char *buf_start = mmap(NULL, n_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
#endif
    char *buf_end = buf_start + n_bytes;

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
    int *accum = calloc(MAX_VALUE, sizeof(int));
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
    }
    thr_handle handles[N_THREADS];
    if (!thr_create_threads(N_THREADS, handles,
                            sizeof(parse_chunk_thread_args),
                            args, parse_chunk_thread)) {
        return false;
    }
    if (!thr_wait_for_threads(N_THREADS, handles)) {
        return false;
    }

    uint64_t max = 0;
    for (int i = 0; i < MAX_VALUE; i++) {
        uint64_t val = accum[i] * i;
        if (val > max) {
            max = val;
        }
    }
    uint64_t time_parsed = nano_count();

    free(accum);
    #if _WIN32
    free(buf_start);
    #else
    if (munmap(buf_start, n_bytes) == -1) {
        return false;
    }
    #endif

    // Print timings.
    double read_secs = NS_TO_S(time_read - time_start);
    double parse_secs = NS_TO_S(time_parsed - time_read);
    double total_secs = NS_TO_S(time_parsed - time_start);
    printf("Read  : %.3f seconds\n", read_secs);
    printf("Parse : %.3f seconds\n", parse_secs);
    printf("Total : %.3f seconds\n", total_secs);
    printf("-- Max: %" PRIu64 "\n", max);
    return true;
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
        if (!run_test(argv[2])) {
            printf("Test run failed!\n");
            return EXIT_FAILURE;
        }
    } else {
        printf("%s: [generate|test] path\n", argv[0]);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
