#include <assert.h>
#include "collectors/ref-counting.h"

void
test_do_allot() {
    ptr mem = (ptr)malloc(4096);

    ref_counting_gc *rc = rc_init(mem, 4096);

    ptr p = rc_do_allot(rc, TYPE_INT, 16);
    assert(mem <= p && p < (mem + 4096));
    assert(rc_space_used(rc) == 16);

    rc_free(rc);

    free((void*)mem);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_do_allot);
    return 0;
}
