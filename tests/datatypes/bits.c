#include <assert.h>
#include <stdio.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"

void
test_mask() {
    printf("0x%x\n", BF_BASIC_MASK(16));
    printf("0x%x\n", BF_MASK(8, 24));

    ptr m = BF_SET(m, 33, 8, 24);
    assert(BF_GET(m, 8, 24) == 33);
    m = BF_SET(m, 100, 0, 8);
    assert(BF_GET(m, 8, 24) == 33);
    assert(BF_GET(m, 0, 8) == 100);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_mask);
}
