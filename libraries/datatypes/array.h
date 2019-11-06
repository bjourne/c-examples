// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#ifndef ARRAY_BL_H
#define ARRAY_BL_H

#include <stdio.h>

// Things you can do with arrays. For now only shuffling and sorting.
void array_shuffle(void *array, size_t n, size_t size);

// Quicksorting with a custom key function.
void array_qsort_with_key(void *base, size_t nmemb, size_t size,
                          int (*key_fun)(void *ctx, const void *item),
                          void *ctx);


#endif
