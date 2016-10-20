#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "collectors/vm.h"
#include "collectors/mark-sweep.h"
#include "collectors/copying.h"
#include "collectors/copying-opt.h"

gc_dispatch *dispatch = NULL;

ptr random_object(vm* v) {
    if (rand_n(2)) {
        return vm_array_init(v, rand_n(200), random_object(v));
    } else {
        switch (rand() % TYPE_ARRAY) {
        case TYPE_INT:
            return vm_boxed_int_init(v, rand_n(100));
        case TYPE_FLOAT:
            return vm_boxed_float_init(v, (double)rand_n(100));
        case TYPE_WRAPPER:
            return vm_wrapper_init(v, random_object(v));
        default:
            return 0;
        }
    }
}

void
test_vm() {
    vm *v = vm_init(dispatch, 200);
    assert(vm_size(v) == 0);
    vm_add(v, 0);
    assert(vm_size(v) == 1);
    assert(vm_remove(v) == 0);
    assert(vm_size(v) == 0);
    vm_free(v);
}

void
test_collect() {
    vm *v = vm_init(dispatch, 400);
    ptr p = vm_add(v, vm_boxed_int_init(v, 29));
    assert(P_GET_TYPE(p) == TYPE_INT);
    assert(vm_space_used(v) == NPTRS(2));
    vm_remove(v);
    vm_collect(v);
    assert(vm_space_used(v) == 0);

    vm_add(v, vm_boxed_int_init(v, 30));
    vm_add(v, vm_boxed_int_init(v, 25));
    assert(vm_space_used(v) == NPTRS(4));
    vm_remove(v);
    vm_collect(v);
    assert(vm_space_used(v) == NPTRS(2));

    vm_add(v, vm_array_init(v, 10, 0));
    assert(vm_space_used(v) == NPTRS(2 + 12 + 2));
    vm_remove(v);
    vm_remove(v);
#if defined(REF_COUNTING_NORMAL) || defined(REF_COUNTING_CYCLES)
    assert(vm_space_used(v) == 0);
#endif
    vm_free(v);
}

void
test_dump() {
    vm *v = vm_init(dispatch, 400);
    vm_add(v, vm_array_init(v, 10, 0));

    vm_add(v, vm_boxed_int_init(v, 20));
    vm_collect(v);
    vm_remove(v);
    vm_free(v);
}

void
test_stack_overflow() {
    vm *v = vm_init(dispatch, 10 << 20);
    vm_add(v, vm_wrapper_init(v, 0));
    size_t n_loops = 300000;
    for (int i = 0; i < n_loops; i++) {
        vm_set(v, 0, vm_wrapper_init(v, vm_get(v, 0)));
    }
    vm_collect(v);
    assert(vm_space_used(v) == (n_loops + 1) * NPTRS(2));
    vm_remove(v);
    vm_collect(v);
    assert(vm_space_used(v) == 0);
    vm_free(v);
}

void
test_mark_stack_overflow() {
    vm *v = vm_init(dispatch, 2048);
    vm_add(v, vm_array_init(v, 20, vm_boxed_int_init(v, 20)));
    vm_collect(v);
    vm_add(v, vm_array_init(v, 40, 0));
    vm_collect(v);
    vm_free(v);
}

void
test_random_stuff() {
    vm *v = vm_init(dispatch, 2048);

    ptr bf = vm_boxed_float_init(v, 2.731);
    vm_add(v, bf);
    vm_add(v, vm_boxed_int_init(v, 33));

    ptr w = vm_wrapper_init(v, bf);
    vm_add(v, w);
    w = vm_wrapper_init(v, w);
    vm_add(v, w);
    w = vm_wrapper_init(v, w);
    vm_add(v, w);
    w = vm_wrapper_init(v, w);
    vm_add(v, w);
    w = vm_wrapper_init(v, w);
    vm_add(v, w);

    vm_set(v, 0, vm_array_init(v, 10, 0));
    vm_set(v, 1, vm_array_init(v, 10, 0));
    vm_set_slot(v, vm_get(v, 1), 1, vm_boxed_int_init(v, 99));
    vm_set_slot(v, vm_get(v, 1), 2, vm_get(v, 0));

    for (size_t n = 0; n < 3000; n++) {
        vm_set(v, n % 4, vm_array_init(v, 7, 0));
    }
    vm_set(v, 0, 0);
    vm_tree_dump(v);
    vm_collect(v);
    vm_tree_dump(v);
    vm_free(v);
}

void
test_torture() {
    vm *v = vm_init(dispatch, 1 << 27);
    for (size_t i = 0; i < 100; i++) {
        vm_add(v, vm_array_init(v, 500, 0));
    }
    for (size_t i = 0; i < 400000; i++) {
        ptr rand_arr = vm_get(v, rand_n(vm_size(v)));
        size_t n_els = *SLOT_P(*SLOT_P(rand_arr, 0), 0);
        ptr p = random_object(v);
        vm_set_slot(v, rand_arr, 1 + rand_n(n_els), p);
    }
    vm_collect(v);
    vm_free(v);
}

int
main(int argc, char *argv[]) {
    srand(time(NULL));
    gc_dispatch *dispatches[3] = {
        ms_get_dispatch_table(),
        cg_get_dispatch_table(),
        cg_get_dispatch_table_optimized()
    };
    char *names[3] = {
        "Mark & Sweep",
        "Copying",
        "Optimized Copying"
    };
    for (size_t n = 0; n < 2; n++) {
        printf("== Running the %s Collector ==\n", names[n]);
        dispatch = dispatches[n];
        PRINT_RUN(test_vm);
        PRINT_RUN(test_collect);
        PRINT_RUN(test_dump);
        PRINT_RUN(test_stack_overflow);
        PRINT_RUN(test_mark_stack_overflow);
        PRINT_RUN(test_random_stuff);
        PRINT_RUN(test_torture);
    }
    return 0;
}
