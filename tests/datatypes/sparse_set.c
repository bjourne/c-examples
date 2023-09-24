// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "datatypes/gens_set.h"
#include "datatypes/sparse_set.h"
#include "random/random.h"

void
test_sparse_set() {
    sparse_set *ss = sparse_set_init(1000);
    assert(!sparse_set_contains(ss, 123));
    sparse_set_add(ss, 123);
    assert(ss->used == 1);
    assert(sparse_set_contains(ss, 123));
    assert(!sparse_set_remove(ss, 321));
    assert(sparse_set_contains(ss, 123));

    sparse_set_add(ss, 5);
    assert(sparse_set_remove(ss, 123));
    assert(!sparse_set_contains(ss, 123));
    assert(sparse_set_contains(ss, 5));

    sparse_set_clear(ss);
    assert(sparse_set_add(ss, 500));

    sparse_set_free(ss);
}

void
test_gens_set() {
    gens_set *gs = gens_set_init(5000);
    assert(!gens_set_contains(gs, 1234));

    assert(gens_set_add(gs, 1234));
    assert(!gens_set_add(gs, 1234));
    assert(gens_set_remove(gs, 1234));
    assert(!gens_set_remove(gs, 1234));

    assert(gens_set_add(gs, 10));
    assert(gens_set_contains(gs, 10));
    gens_set_clear(gs);
    assert(!gens_set_contains(gs, 10));

    gens_set_free(gs);
}

#define SET_SIZE (1 * 1000 * 1000)

void
benchmark_sparse_set_add() {
    sparse_set *ss = sparse_set_init(SET_SIZE);
    for (size_t i = 0; i < 10 * SET_SIZE; i++) {
        sparse_set_add(ss, rnd_pcg32_rand_range(SET_SIZE));
    }
    sparse_set_free(ss);
}

void
benchmark_gens_set_add() {
    gens_set *gs = gens_set_init(SET_SIZE);
    for (size_t i = 0; i < 10 * SET_SIZE; i++) {
        gens_set_add(gs, rnd_pcg32_rand_range(SET_SIZE));
    }
    gens_set_free(gs);
}


int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_sparse_set);
    PRINT_RUN(test_gens_set);
    PRINT_RUN(benchmark_sparse_set_add);
    PRINT_RUN(benchmark_gens_set_add);
    return 0;
}
