// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
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
    // This has to be fixed because it doesn't work if the elements
    // are, say, structs.
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

typedef struct {
    void *ctx;
    array_cmp_fun *fun;
    void *base;
    size_t size;
} indirect_compare_context;

static int
ind_cmp_fun(const void *a, const void *b, void *ctx) {
    indirect_compare_context *wrap = (indirect_compare_context *)ctx;
    size_t ai = (size_t)a & 0xffffffff;
    size_t bi = (size_t)b & 0xffffffff;
    void *av = wrap->base + wrap->size * ai;
    void *bv = wrap->base + wrap->size * bi;
    return wrap->fun(*(void **)av, *(void **)bv, wrap->ctx);
}

size_t*
array_qsort_indirect(void *base, size_t nmemb, size_t size,
                     array_cmp_fun *fun, void *ctx) {
    size_t *indices = malloc(nmemb * sizeof(size_t));
    for (size_t i = 0; i < nmemb; i++) {
        indices[i] = i;
    }
    indirect_compare_context wrap = { ctx, fun, base, size };
    array_qsort(indices, nmemb, sizeof(size_t),
                ind_cmp_fun, (void *)&wrap);
    return indices;
}


void
array_permute(void *base, size_t nmemb, size_t size, size_t *indices) {
    for (size_t i = 0; i < nmemb - 1; i++) {
        size_t ind = indices[i];
        while (ind < i) {
            ind = indices[ind];
        }
        if (size == 1) {
            uint8_t *base8 = (uint8_t *)base;
            uint8_t tmp = base8[i];
            base8[i] = base8[ind];
            base8[ind] = tmp;
        } else if (size == 4) {
            uint32_t *base32 = (uint32_t *)base;
            uint32_t tmp = base32[i];
            base32[i] = base32[ind];
            base32[ind] = tmp;
        } else if (size == 8) {
            uint64_t *base64 = (uint64_t *)base;
            uint64_t tmp = base64[i];
            base64[i] = base64[ind];
            base64[ind] = tmp;
        } else {
            assert(false);
        }
    }
}

int
array_ord_asc_u8(const void *a,
                 const void *b,
                 void *ctx __attribute__((unused))) {
    uint8_t ai = (uint8_t)((uintptr_t)a & 0xff);
    uint8_t bi = (uint8_t)((uintptr_t)b & 0xff);
    return ai - bi;
}

int
array_ord_asc_u32(const void *a,
                  const void *b,
                  void *ctx __attribute__((unused))) {
    uint32_t ai = (uint32_t)((uintptr_t)a & 0xffffffff);
    uint32_t bi = (uint32_t)((uintptr_t)b & 0xffffffff);
    if (ai > bi) {
        return 1;
    } else if (ai < bi) {
        return -1;
    }
    return 0;
}

int
array_ord_asc_i32(const void *a,
                  const void *b,
                  void *ctx __attribute__((unused))) {
    int32_t ai = (int32_t)((uintptr_t)a & 0xffffffff);
    int32_t bi = (int32_t)((uintptr_t)b & 0xffffffff);
    return ai - bi;
}

static int32_t *
fast_bsearch_i32(int32_t *base, size_t nmemb, int32_t key) {
    while (nmemb) {
        size_t half = nmemb >> 1;
        if (base[half] < key) {
            base += nmemb - half;
        }
        nmemb = half;
    }
    return base;
}

static uint32_t *
fast_bsearch_u32(uint32_t *base, size_t nmemb, uint32_t key) {
    while (nmemb) {
        size_t half = nmemb >> 1;
        __builtin_prefetch(&base[half >> 1]);
        uint32_t *base_half = base + (nmemb - half);
        __builtin_prefetch(&base_half[half >> 1]);
        base = base[half] < key ? base_half : base;
        nmemb = half;
    }
    return base;
}

#define BSEARCH_BODY(expr)                          \
    while (nmemb > 0) {                             \
        size_t rem = nmemb & 1;                     \
        nmemb >>= 1;                                \
        if (expr) {                                 \
            first += nmemb + rem;                   \
        }                                           \
    }

size_t
array_bsearch(void *base, size_t nmemb, size_t size,
              array_cmp_fun *fun, void *ctx,
              void *key) {
    size_t first = 0;
    if (size == 4) {
        if (fun == array_ord_asc_i32) {
            int32_t key32 = (int32_t)((uint64_t)key);
            first = fast_bsearch_i32(base, nmemb, key32)
                - (int32_t *)base;
        } else if (fun == array_ord_asc_u32) {
            uint32_t key32 = (uint32_t)((uint64_t)key);
            first = fast_bsearch_u32(base, nmemb, key32)
                - (uint32_t *)base;
        } else {
            BSEARCH_BODY(fun(*(void **)(base + 4 * (first + nmemb)),
                             key, ctx) < 0);
        }
    } else {
        BSEARCH_BODY(fun(*(void **)(base + size * (first + nmemb)),
                         key, ctx) < 0);
    }
    return first;
}
