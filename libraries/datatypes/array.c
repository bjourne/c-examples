// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
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
    int (*key_fun)(void *ctx, const void *a);
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
    int k1 = outer->key_fun(outer->ctx, a);
    int k2 = outer->key_fun(outer->ctx, b);
    return k1 - k2;
}

void
array_qsort_with_key(void *base, size_t nmemb, size_t size,
                     int (*key_fun)(void *ctx, const void *a),
                     void *ctx) {
    compare_context outer_ctx = { ctx, key_fun };
#if defined(_MSC_VER)
    qsort_s(base, nmemb, size, cmp_fun, (void *)&outer_ctx);
#else
    qsort_r(base, nmemb, size, cmp_fun, (void *)&outer_ctx);
#endif
}
