// Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <inttypes.h>
#include <time.h>
#include "datatypes/hashset.h"

void
test_bad_bug() {
    hashset *hs = hs_init();
    int values[] = {
        3, 35, 67, 99, 131, 163, 195,
        227, 259, 291, 323, 355, 387
    };
    for (int i = 0; i < ARRAY_SIZE(values); i++) {
        hs_add(hs, values[i]);
        assert(hs_in_p(hs, values[i]));
    }
    hs_free(hs);
}

void
test_print_stuff() {
    hashset* hs = hs_init();
    for (int i = 0; i < 10; i++) {
        size_t el = rand_n(100);
        hs_add(hs, el);
    }
    HS_FOR_EACH_ITEM(hs, { printf("%" PRIuPTR "\n", p); });
    hs_free(hs);
}

void
test_adding_reserved() {
    hashset *hs = hs_init();
    for (int i = 0; i < 2; i++) {
        assert(!hs_add(hs, i));
        assert(!hs_in_p(hs, i));
    }
    hs_free(hs);
}

void
test_adding_and_deleting() {
    hashset *hs = hs_init();
    for (int i = 0; i < 1000; i++) {
        uint64_t v = rand_n(0xffffff) + 2;
        hs_add(hs, v);
        assert(hs_in_p(hs, v));
        assert(hs_remove(hs, v));
        assert(!hs_in_p(hs, v));
    }
    hs_free(hs);
}


void
test_next_key() {
    hashset *hs = malloc(sizeof(hashset));
    hs->capacity = HS_INITIAL_CAPACITY << 10;
    hs->mask = hs->capacity - 1;
    hs->array = calloc(hs->capacity, sizeof(size_t));
    hs->n_items = 0;
    hs->n_used = 0;

    for (int j = 2; j < 10000; j++) {
        int* seen = (int *)calloc(hs->capacity, sizeof(int));
        int at = HS_FIRST_KEY(hs, j);
        int n_tries = hs->capacity / 2;
        for (int i = 0; i < n_tries; i++) {
            assert(seen[at] == 0);
            seen[at] = 1;
            at = HS_NEXT_KEY(hs, at);
        }
        free(seen);
    }
}


int
main(int argc, char *argv[]) {

    rand_init(0);

    PRINT_RUN(test_bad_bug);
    PRINT_RUN(test_print_stuff),
    PRINT_RUN(test_adding_reserved);
    PRINT_RUN(test_adding_and_deleting);
    PRINT_RUN(test_next_key);

    return 0;
}
