// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#ifndef ARRAY_BL_H
#define ARRAY_BL_H

#include <stdio.h>

// Things you can do with arrays. For now only shuffling and sorting.
void array_shuffle(void *array, size_t n, size_t size);

typedef int (array_cmp_fun)(const void *a, const void *b, void *ctx);
typedef int (array_key_fun)(const void *a, void *ctx);

void
array_qsort(void *base, size_t nmemb, size_t size,
            int (*cmp_fun)(const void *a, const void *b, void *ctx),
            void *ctx);

// Quicksorting with a custom key function.
void
array_qsort_by_key(void *base, size_t nmemb, size_t size,
                   array_key_fun *fun, void *ctx);


#endif
