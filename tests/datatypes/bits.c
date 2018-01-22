#include <assert.h>
#include <stdio.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"

void
test_mask() {
    printf("0x%lx\n", BF_LIT_BITS(16));
    printf("0x%lx\n", BF_MASK(8, 24));

    ptr m = BF_MERGE(0, 33, 8, 24);
    assert(BF_GET(m, 8, 24) == 33);
    m = BF_MERGE(m, 100, 0, 8);
    assert(BF_GET(m, 8, 24) == 33);
    assert(BF_GET(m, 0, 8) == 100);

    m = 0xffffff;
    m = BF_MERGE(m, 2, 0, 4);
    m = BF_MERGE(m, 4, 5, 3);
    assert(BF_GET(m, 5, 3) == 4);
    assert(BF_GET(m, 0, 4) == 2);
}

void
test_float_bits() {
    float floats[2] = {1.0f, -1.0f};
    assert(BW_FLOAT_TO_UINT(floats[0]) == 0x3f800000);
    assert(BW_FLOAT_TO_UINT(floats[1]) == 0xbf800000);
}

void
test_bit_count() {
    assert(BIT_COUNT(0) == 0);
    assert(BIT_COUNT(1) == 1);
    assert(BIT_COUNT(2) == 1);
    assert(BIT_COUNT(3) == 2);
    assert(BIT_COUNT(0xffffffffffffffff) == 64);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_mask);
    PRINT_RUN(test_float_bits);
    PRINT_RUN(test_bit_count);
}
