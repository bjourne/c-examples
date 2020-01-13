// N THREADS  VARIANT           TIME
// ---------  ---------------  -----
//         8  interlocked add  2.978
//         6  interlocked add  2.954
//         5  interlocked add  2.978
//         4  interlocked add  2.960
//         3  interlocked add  3.040
//         2  interlocked add  3.401
//         3  multimem         3.106
//         4  multimem         3.118
//         5  multimem         3.198
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "datatypes/common.h"

#define MIN_VALUE (1 * 1000 * 1000)
#define MAX_VALUE (2 * 1000 * 1000)
#define N_VALUES 100 * 1000 * 1000
#define N_THREADS 6

#define NANO_TO_SEC(x)      ((double)(x) / 1000 / 1000 / 1000)

static int
rand_in_range(int lo, int hi) {
    int range = hi - lo;
    int val = (rand() & 0xff) << 16 |
        (rand() & 0xff) << 8 |
        (rand() & 0xff);
    return (val % range) + lo;
}

static void
run_generate() {
    srand(1234);
    FILE *f = fopen("data.txt", "wb");
    for (int i = 0; i < N_VALUES; i++) {
        fprintf(f, "%d\n", rand_in_range(MIN_VALUE, MAX_VALUE));
    }
    fclose(f);
}

// Macros for loop unrolling.
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
    size_t val;
    while (buf < end) {
        // Parse up to 7 digits.xs
        PARSE_FIRST_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
        PARSE_NEXT_DIGIT;
    done:
        InterlockedExchangeAdd64(&accum[val], val);
        // Skip newline character.
        buf++;
    }
}

typedef struct {
    char *chunk_start;
    char *chunk_end;
    size_t *accum;
} parse_chunk_thread_args;

static DWORD WINAPI
parse_chunk_thread(LPVOID args) {
    parse_chunk_thread_args *a = (parse_chunk_thread_args *)args;
    parse_chunk(a->chunk_start, a->chunk_end, a->accum);
    return 0;
}

static void
run_test() {
    size_t time_start = nano_count();

    FILE *f = fopen("data.txt", "rb");
    fseek(f, 0, SEEK_END);
    long int n_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf_start = (char *)malloc(sizeof(char) * n_bytes);
    char *buf_end = buf_start + n_bytes;
    assert(fread(buf_start, 1, n_bytes, f) == n_bytes);
    fclose(f);

    size_t time_read = nano_count();

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

    HANDLE threads[N_THREADS];
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
        threads[i] = CreateThread(NULL, 0, parse_chunk_thread,
                                  &args[i], 0, NULL);
    }
    for (int i = 0; i < N_THREADS; i++) {
        WaitForSingleObject(threads[i], INFINITE);
    }

    size_t time_parsed = nano_count();
    size_t max = 0;
    for (int i = 0; i < MAX_VALUE; i++) {
        size_t val = accum[i];
        if (val > max) {
            max = val;
        }
    }

    size_t time_accum = nano_count();

    printf("max = %zu\n", max);
    free(accum);
    free(buf_start);

    // Print timings.
    double read_secs = NANO_TO_SEC(time_read - time_start);
    double parse_secs = NANO_TO_SEC(time_parsed - time_read);
    double accum_secs = NANO_TO_SEC(time_accum - time_parsed);
    double total_secs = NANO_TO_SEC(time_accum - time_start);
    printf("Read : %.3f seconds\n", read_secs);
    printf("Parse: %.3f seconds\n", parse_secs);
    printf("Accum: %.3f seconds\n", accum_secs);
    printf("Total: %.3f seconds\n", total_secs);
}

int
main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("%s: generate/test\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *cmd = argv[1];
    if (strcmp(cmd, "generate") == 0) {
        run_generate();
    } else if (strcmp(cmd, "test") == 0) {
        run_test();
    }
    return EXIT_SUCCESS;
}
