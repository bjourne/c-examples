// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef ARRAY_BL_H
#define ARRAY_BL_H

// To smooth over differences between qsort_s and qsort_r.
typedef int (array_cmp_fun)(const void *a, const void *b, void *ctx);
typedef int (array_key_fun)(const void *a, void *ctx);

void
array_qsort(void *base, size_t nmemb, size_t size,
            array_cmp_fun *fun, void *ctx);

size_t*
array_qsort_indirect(void *base, size_t nmemb, size_t size,
                     array_cmp_fun *fun, void *ctx);

// Quicksorting with a custom key function.
void
array_qsort_by_key(void *base, size_t nmemb, size_t size,
                   array_key_fun *fun, void *ctx);

// Builtin sorts
int array_ord_asc_u8(const void *a, const void *b, void *ctx);
int array_ord_asc_u32(const void *a, const void *b, void *ctx);
int array_ord_asc_i32(const void *a, const void *b, void *ctx);

void array_permute(void *base, size_t nmemb, size_t size, size_t *indices);
void array_shuffle(void *array, size_t n, size_t size);


// Binary search. Returns the insertion point.
size_t
array_bsearch(void *base, size_t nmemb, size_t size,
              array_cmp_fun *fun, void *ctx,
              void *key);


#endif
