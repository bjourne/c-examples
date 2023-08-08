// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/bitarray.h"

void
test_twiddling() {
    assert(rightmost_clear_bit(1) == 1);
    assert(rightmost_clear_bit(3) == 2);
    assert(rightmost_clear_bit(7) == 3);
    assert(rightmost_clear_bit(5) == 1);
}

void
test_basic() {
    bitarray *ba = ba_init(64);
    assert(ba->n_words == 1);
    ba_free(ba);

    ba = ba_init(128);
    assert(ba->n_words == 2);
    ba_free(ba);
}

void
test_set_and_get_bit() {
    bitarray *ba = ba_init(128);
    assert(ba->n_words == 2);
    ba_set_bit(ba, 0);
    assert(AT(ba->bits) == 1);
    ba_set_bit(ba, 1);
    assert(AT(ba->bits) == 3);

    assert(ba_get_bit(ba, 0));

    assert(!ba_get_bit(ba, 58));
    ba_set_bit(ba, 58);
    assert(ba_get_bit(ba, 58));

    ba_free(ba);
}

void
test_set_range() {
    bitarray *ba = ba_init(1024);
    assert(ba->n_words == 16);

    ba_set_bit_range(ba, 8, 0);
    assert(!ba_get_bit(ba, 8));

    ba_set_bit_range(ba, 12, 3);

    assert(ba_get_bit(ba, 12));
    assert(ba_get_bit(ba, 13));
    assert(ba_get_bit(ba, 14));
    assert(!ba_get_bit(ba, 15));

    ba_free(ba);

    ba = ba_init(1024);
    assert(ba->n_words == 16);
    ba_set_bit_range(ba, 0, 1024);
    for (int i = 0; i < 1024; i++) {
        assert(ba_get_bit(ba, i));
    }
    ba_clear(ba);
    ba_set_bit_range(ba, 10, 112);
    assert(AT(ba->bits) == 0xfffffffffffffc00);
    assert(AT(ba->bits + sizeof(ptr)) == 0x3ffffffffffffff);
    for (int i = 0; i < 112; i++) {
        assert(ba_get_bit(ba, 10 + i));
    }

    ba_clear(ba);
    ba_set_bit_range(ba, 0, 3);
    assert(AT(ba->bits) == 7);

    ba_free(ba);
}

void
test_next_unset_bit() {
    bitarray *ba = ba_init(1024);
    ba_set_bit_range(ba, 0, 65);
    assert(AT(ba->bits) == 0xffffffffffffffff);
    assert(AT(ba->bits + sizeof(ptr)) == 1);

    assert(ba_next_unset_bit(ba, 0) == 65);
    assert(ba_get_bit(ba, 64));
    assert(!ba_get_bit(ba, 65));


    assert(ba_next_unset_bit(ba, 65) == 65);

    ba_set_bit(ba, 65);

    assert(ba_next_unset_bit(ba, 65) == 66);

    ba_clear(ba);

    assert(ba_next_unset_bit(ba, 0) == 0);
    ba_set_bit_range(ba, 0, 1024);
    assert(ba_next_unset_bit(ba, 0) == 1024);

    ba_set_bit(ba, 20);
    assert(ba_next_unset_bit(ba, 20) == 1024);

    ba_free(ba);
}

void
test_next_set_bit() {
    bitarray *ba = NULL;
    ba = ba_init(1024);
    assert(ba_next_set_bit(ba, 0) == 1024);
    ba_set_bit(ba, 0);
    assert(ba_next_set_bit(ba, 0) == 0);
    assert(ba_next_unset_bit(ba, 0) == 1);


    ba_set_bit_range(ba, 100, 20);
    assert(ba_next_set_bit(ba, 10) == 100);

    ba_free(ba);

    ba = ba_init(64);
    ba_clear(ba);
    assert(ba_next_set_bit(ba, 0) == 64);
    ba_set_bit_range(ba, 0, 64);
    assert(ba_next_unset_bit(ba, 0) == 64);
    ba_free(ba);

    ba = ba_init(640);
    ba_set_bit(ba, 400);
    int v = ba_next_set_bit(ba, 350);

    assert(v == 400);
    ba_free(ba);
}

void
test_bitsum() {
    bitarray *ba = ba_init(640);

    ba_clear(ba);
    assert(ba_bitsum(ba) == 0);
    ba_set_bit(ba, 12);
    assert(ba_bitsum(ba) == 1);

    ba_clear(ba);
    ba_set_bit_range(ba, 0, 11);
    assert(ba_bitsum(ba) == 11);

    ba_clear(ba);
    ba_set_bit_range(ba, 40, 11);
    assert(ba_bitsum(ba) == 11);

    ba_clear(ba);
    ba_set_bit_range(ba, 127, 1);
    assert(ba_bitsum(ba) == 1);
    ba_clear(ba);

    ba_set_bit_range(ba, 119, 11);
    ba_set_bit_range(ba, 62, 1);
    ba_set_bit_range(ba, 63, 1);
    assert(ba_bitsum(ba) == 13);

    ba_clear(ba);
    ba_set_bit_range(ba, 100, 250);
    assert(ba_bitsum(ba) == 250);
    assert(ba_next_unset_bit(ba, 0) == 0);
    assert(ba_next_set_bit(ba, 0) == 100);
    assert(ba_next_unset_bit(ba, 100) == 350);

    ba_set_bit_range(ba, 400, 1);
    assert(ba_get_bit(ba, 400));
    assert(ba_bitsum(ba) == 251);
    assert(!ba_get_bit(ba, 350));
    assert(ba_next_set_bit(ba, 370) == 400);

    int count = 0;
    BA_EACH_UNSET_RANGE(ba, {
        assert(addr >= 0);
        assert(size > 0);
        count++;
    });
    assert(count == 3);
    ba_free(ba);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_twiddling);
    PRINT_RUN(test_basic);
    PRINT_RUN(test_set_and_get_bit);
    PRINT_RUN(test_set_range);
    PRINT_RUN(test_next_unset_bit);
    PRINT_RUN(test_next_set_bit);
    PRINT_RUN(test_bitsum);
    return 0;
}
