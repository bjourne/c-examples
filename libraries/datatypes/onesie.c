#include "datatypes/onesie.h"

onesie *
os_init(size_t n, size_t size) {
    onesie *os = (onesie *)malloc(sizeof(onesie));
    os->region = (ptr)malloc(n * size);
    os->free_blocks = v_init(n);

    ptr addr = os->region;
    while (n) {
        v_add(os->free_blocks, addr);
        addr += size;
        n--;
    }
    return os;
}

ptr
os_allot_block(onesie *os) {
    return v_remove(os->free_blocks);
}

void
os_free_block(onesie *os, ptr block) {
    v_add(os->free_blocks, block);
}

bool
os_can_allot_p(onesie *os) {
    return os->free_blocks->used > 0;
}

void
os_free(onesie *os) {
    v_free(os->free_blocks);
    free((void *)os->region);
    free(os);
}
