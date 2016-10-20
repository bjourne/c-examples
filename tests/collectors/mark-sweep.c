#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "collectors/vm.h"

void test_ms_collect() {
    vm *v = vm_init(400);
    ptr p = vm_add(v, vm_boxed_int_init(v, 29));
    assert(P_GET_TYPE(p) == TYPE_INT);
    assert(ms_space_used(v->mem_man) == NPTRS(2));
    vm_remove(v);
    ms_collect(v->mem_man);
    assert(ms_space_used(v->mem_man) == 0);

    vm_add(v, vm_boxed_int_init(v, 30));
    vm_add(v, vm_boxed_int_init(v, 25));
    assert(ms_space_used(v->mem_man) == NPTRS(4));
    vm_remove(v);
    ms_collect(v->mem_man);
    assert(ms_space_used(v->mem_man) == NPTRS(2));

    vm_add(v, vm_array_init(v, 10, 0));
    assert(ms_space_used(v->mem_man) == NPTRS(2 + 12 + 2));
    vm_remove(v);
    vm_remove(v);
#if defined(REF_COUNTING_NORMAL) || defined(REF_COUNTING_CYCLES)
    assert(ms_space_used(v->mem_man) == 0);
#endif
    vm_free(v);
}

int
main(int argc, char *argv[]) {
    srand(time(NULL));
    PRINT_RUN(test_ms_collect);
    return 0;
}
