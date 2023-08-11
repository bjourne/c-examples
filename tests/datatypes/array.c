// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "datatypes/array.h"
#include "datatypes/common.h"
#include "random/random.h"

void
test_array_shuffle() {
    int orig[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int new[10];
    memcpy(new, orig, sizeof(int) * 10);
    array_shuffle(new, 10, sizeof(int));
    bool same = true;
    for (int i = 0; i < 10; i++) {
        same = same && (orig[i] == new[i]);
    }
    // The odds are low...
    assert(!same);
    for (int i = 0; i < 10; i++) {
        printf("%d\n", new[i]);
    }
}

typedef struct {
    int *values;
} sort_context;

static int
key_fun(const void *a, void *ctx) {
    sort_context *s_ctx = (sort_context *)ctx;
    int idx = (int)((uintptr_t)a & 0xffffffff);
    return s_ctx->values[idx];
}

void
test_sorting() {
    int items[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int values[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    sort_context ctx = { values };
    array_qsort_by_key(items, 10, sizeof(int), &key_fun, &ctx);
    for (int i = 0; i < 10; i++) {
        printf("%d ", items[i]);
    }
    printf("\n");
}

void
test_sorting_strings() {
    char *strings[] = {
        "one",
        "two",
        "three",
        "foo"
    };
    array_qsort(strings, 4, sizeof(char *), (array_cmp_fun *)&strcmp, NULL);
    assert(!strcmp(strings[0], "foo"));
    assert(!strcmp(strings[1], "one"));
    assert(!strcmp(strings[2], "three"));
    assert(!strcmp(strings[3], "two"));

    // By string length
    array_qsort_by_key(strings, 4, sizeof(char *), (array_key_fun *)&strlen, NULL);
    assert(!strcmp(strings[0], "foo"));
    assert(!strcmp(strings[1], "one"));
    assert(!strcmp(strings[2], "two"));
    assert(!strcmp(strings[3], "three"));
}

void
test_argsort() {
    char *strings[] = {
        "foo",
        "bar",
        "xyz",
        "aaa",
        "nnnn"
    };
    size_t *indices = array_qsort_indirect(strings, 5, sizeof(char *),
                                           (array_cmp_fun *)&strcmp, NULL);
    for (int i = 0; i < 5; i++) {
        printf("%ld\n", indices[i]);
    }
    free(indices);
}

void
test_permute() {
    char *strings[] = {
        "a", "b", "c", "d",
        "e", "f", "g", "h"
    };
    char *expected[] = {
        "a", "c", "b", "d",
        "e", "h", "g", "f"
    };
    size_t indices[8] = {0, 2, 1, 3, 4, 7, 6, 5};
    array_permute(strings, ARRAY_SIZE(indices),
                  sizeof(char *), indices);

    for (size_t i = 0; i < ARRAY_SIZE(indices); i++) {
        assert(!strcmp(strings[i], expected[i]));
    }
}

void
test_permute2() {
    uint8_t delays[] = {
        19, 7, 22, 29, 15,
        20, 17, 21, 18, 19
    };
    size_t n_delays = ARRAY_SIZE(delays);
    size_t *idxs = array_qsort_indirect(
        delays, n_delays,
        sizeof(uint8_t),
        (array_cmp_fun *)&array_ord_asc_u8,
        NULL);

    size_t exp_idxs[10] = {
        1, 4, 6, 8, 0,
        9, 5, 7, 2, 3
    };
    for (size_t i = 0; i < n_delays; i++) {
        assert(exp_idxs[i] == idxs[i]);
    }

    // Now apply the permutation
    array_permute(delays, n_delays, sizeof(uint8_t), idxs);

    uint8_t exp_delays[10] = {
        7, 15, 17, 18, 19, 19, 20, 21, 22, 29
    };
    for (size_t i = 0; i < n_delays; i++) {
        assert(exp_delays[i] == delays[i]);
    }
    free(idxs);
}

void
test_bsearch() {
    int arr1[] = {3, 4, 9, 10, 15, 20, 55};
    size_t nmemb1 = ARRAY_SIZE(arr1);
    size_t size1 = sizeof(int);

    assert(array_bsearch(arr1, nmemb1, size1, array_ord_asc_i32, NULL, (void *)15) == 4);
    assert(array_bsearch(arr1, nmemb1, size1, array_ord_asc_i32, NULL, (void *)100) == 7);
    assert(array_bsearch(arr1, nmemb1, size1, array_ord_asc_i32, NULL, (void *)-10) == 0);

    int arr2[] = {};
    assert(array_bsearch(arr2, 0, size1, &array_ord_asc_i32, NULL, (void *)123) == 0);

    uint8_t arr3[] = {2, 3, 99, 200};
    size_t nmemb3 = ARRAY_SIZE(arr3);
    size_t size3 = sizeof(uint8_t);
    assert(array_bsearch(arr3, nmemb3, size3,
                         &array_ord_asc_u8, NULL, (void *)5) == 2);

    uint32_t arr4[] = {0, 5, 10};
    assert(array_bsearch(arr4, 3, sizeof(uint32_t),
                         &array_ord_asc_u32, NULL, (void *)-1) == 3);
    assert(array_bsearch(arr4, 3, sizeof(uint32_t),
                         &array_ord_asc_u32, NULL, (void *)1) == 1);
}

// See https://mhdm.dev/posts/sb_lower_bound/
#define N_RND_BUF (100 * 1000 * 1000)
#define RND_LIM (1 << 31)
#define N_ACCESSES (10 * 1000 * 1000L)

void
test_bsearch_perf() {
    size_t tp_size = sizeof(uint32_t);
    rnd_pcg32_seed(1001, 370);
    uint32_t *buf = malloc_aligned(64, tp_size * N_RND_BUF);
    rnd_pcg32_rand_range_fill(buf, RND_LIM, N_RND_BUF);
    printf("Sorting\n");
    array_qsort(buf, N_RND_BUF, tp_size, array_ord_asc_u32, NULL);

    uint64_t start = nano_count();
    for (uint64_t i = 0; i < N_ACCESSES; i++) {
        array_bsearch(buf, N_RND_BUF, tp_size,
                      array_ord_asc_u32, NULL, (void *)(uint64_t)i);
    }
    uint64_t nanos = nano_count() - start;
    double nanos_per_access = (double)nanos / (double)N_ACCESSES;
    printf("%.2lfns/access\n", nanos_per_access);
    free(buf);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_array_shuffle);
    PRINT_RUN(test_sorting);
    PRINT_RUN(test_sorting_strings);
    PRINT_RUN(test_argsort);
    PRINT_RUN(test_permute);
    PRINT_RUN(test_permute2);
    PRINT_RUN(test_bsearch);
    PRINT_RUN(test_bsearch_perf);
    return 0;
}
