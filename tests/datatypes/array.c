// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "datatypes/array.h"
#include "datatypes/common.h"

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

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_array_shuffle);
    PRINT_RUN(test_sorting);
    PRINT_RUN(test_sorting_strings);
    return 0;
}
