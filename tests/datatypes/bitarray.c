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
    bitarray *ba = ba_init(13);
    assert(ba->n_words == 1);
    ba_free(ba);

    ba = ba_init(100);
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
    assert(AT(ba->bits) == -1);
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
    assert(ba_next_unset_bit(ba, 0) == -1);

    ba_set_bit(ba, 20);
    assert(ba_next_unset_bit(ba, 20) == -1);

    ba_free(ba);
}

void
test_next_set_bit() {
    bitarray *ba = ba_init(1024);
    assert(ba_next_set_bit(ba, 0) == -1);
    ba_set_bit(ba, 0);
    assert(ba_next_set_bit(ba, 0) == 0);
    assert(ba_next_unset_bit(ba, 0) == 1);


    ba_set_bit_range(ba, 100, 20);
    assert(ba_next_set_bit(ba, 10) == 100);

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
    return 0;
}
