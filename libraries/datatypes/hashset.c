/* -*- Mode: C; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*
 *     Copyright 2012 Couchbase, Inc.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
/* Copyright (C) 2016 Björn Lindqvist

   This file comes originally from
   https://github.com/avsej/hashset.c/blob/master/hashset.c.  It has
   been rewritten to my preferred coding style and has had lots of
   bugs fixed. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "hashset.h"

hashset *hs_init() {
    hashset *hs = malloc(sizeof(hashset));
    hs->capacity = HS_INITIAL_CAPACITY;
    hs->mask = hs->capacity - 1;
    hs->array = calloc(hs->capacity, sizeof(size_t));
    hs->n_items = 0;
    hs->n_used = 0;
    return hs;
}

void hs_free(hashset *hs)
{
    free(hs->array);
    free(hs);
}

void hs_clear(hashset *hs) {
    memset(hs->array, 0, sizeof(size_t) * hs->capacity);
    hs->n_used = 0;
    hs->n_items = 0;
}

static
bool hs_add_member(hashset *hs, size_t item)
{
    if (item <= 1) {
        return false;
    }
    size_t i = HS_FIRST_KEY(hs, item);
    size_t *a = hs->array;
    while (a[i] > 1) {
        if (a[i] == item) {
            return false;
        } else {
            i = HS_NEXT_KEY(hs, i);
        }
    }
    if (a[i] == 0) {
        hs->n_used++;
    }
    hs->n_items++;
    a[i] = item;
    return true;
}

static
void maybe_rehash(hashset *hs)
{
    size_t old_cap = hs->capacity;
    size_t max_used = (size_t)((double)old_cap * HS_MAX_FILL);
    if (hs->n_used >= max_used) {
        size_t *old_array = hs->array;
        hs->capacity *= 2;
        printf("rehashing to %lu\n", hs->capacity);
        hs->mask = hs->capacity - 1;
        hs->array = calloc(hs->capacity, sizeof(size_t));
        hs->n_items = hs->n_used = 0;
        for (size_t ii = 0; ii < old_cap; ii++) {
            hs_add_member(hs, old_array[ii]);
        }
        free(old_array);
    }
}

bool hs_add(hashset *hs, size_t item)
{
    bool rv = hs_add_member(hs, item);
    maybe_rehash(hs);
    return rv;
}

void hs_remove_at(hashset *hs, size_t i) {
    hs->array[i] = 1;
    hs->n_items--;
}

bool hs_remove(hashset *hs, size_t item)
{
    size_t i = HS_FIRST_KEY(hs, item);
    size_t *a = hs->array;
    while (a[i] != 0) {
        if (a[i] == item) {
            hs_remove_at(hs, i);
            return true;
        }
        i = HS_NEXT_KEY(hs, i);
    }
    return false;
}

bool hs_in_p(hashset *hs, size_t item)
{
    size_t i = HS_FIRST_KEY(hs, item);
    while (hs->array[i] != 0) {
        if (hs->array[i] == item) {
            return true;
        } else {
            i = HS_NEXT_KEY(hs, i);
        }
    }
    return false;
}
