// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "array.h"

void
array_shuffle(void *array, size_t n, size_t size) {
    char *arr = array;
    size_t stride = size * sizeof(char);
    char tmp[128];
    assert(size <= 128);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}

typedef struct {
    void *ctx;
    array_cmp_fun *fun;
} compare_context;

static int
cmp_fun(
#if defined (_MSC_VER)
    void *ctx, const void *a, const void *b
#else
    const void *a, const void *b, void *ctx
#endif
) {
    compare_context *outer = (compare_context *)ctx;
    // I don't know if this is kosher...
    return outer->fun(*(void **)a, *(void **)b, outer->ctx);
}

void
array_qsort(void *base, size_t nmemb, size_t size,
            array_cmp_fun *fun, void *ctx) {
    compare_context outer_ctx = { ctx, fun };
#if defined(_MSC_VER)
    qsort_s(base, nmemb, size, cmp_fun, (void *)&outer_ctx);
#else
    qsort_r(base, nmemb, size, cmp_fun, (void *)&outer_ctx);
#endif
}

typedef struct {
    void *ctx;
    array_key_fun *fun;
} key_compare_context;

static int
key_cmp_fun(const void *a, const void *b, void *ctx) {
    key_compare_context *wrap = (key_compare_context *)ctx;
    int k1 = wrap->fun(a, wrap->ctx);
    int k2 = wrap->fun(b, wrap->ctx);
    return k1 - k2;
}

void
array_qsort_by_key(void *base, size_t nmemb, size_t size,
                   array_key_fun *fun, void *ctx) {
    key_compare_context wrap = { ctx, fun };
    array_qsort(base, nmemb, size, key_cmp_fun, (void *)&wrap);
}
