#include <assert.h>
#include "datatypes/onesie.h"

void
test_init_free() {
    onesie *os = os_init(100, 40);
    assert(os->free_blocks->size == 100);
    assert(os->free_blocks->used == 100);
    assert(os->free_blocks->array[0] == os->region);
    os_free(os);
}

void
test_can_allot_p() {
    onesie *os = os_init(0, 40);
    assert(!os_can_allot_p(os));
    os_free(os);

    os = os_init(1, 40);
    assert(os_can_allot_p(os));
    os_free(os);
}

void
test_allot_free(){
    onesie *os = os_init(3, 40);
    ptr b1 = os_allot_block(os);
    assert(os->free_blocks->used == 2);
    ptr b2 = os_allot_block(os);
    assert(os->free_blocks->used == 1);
    ptr b3 = os_allot_block(os);
    assert(os->free_blocks->used == 0);
    assert(!os_can_allot_p(os));
    os_free_block(os, b1);
    assert(os->free_blocks->used == 1);
    os_free_block(os, b2);
    assert(os->free_blocks->used == 2);
    os_free_block(os, b3);
    assert(os->free_blocks->used == 3);
    os_free(os);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_init_free);
    PRINT_RUN(test_can_allot_p);
    PRINT_RUN(test_allot_free);
    return 0;
}
