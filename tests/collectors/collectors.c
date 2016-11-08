#include <assert.h>
#include <stdlib.h>
#include "collectors/vm.h"
#include "collectors/copying.h"
#include "collectors/copying-opt.h"
#include "collectors/mark-sweep.h"
#include "collectors/ref-counting.h"
#include "collectors/ref-counting-cycles.h"

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
    vm *v = vm_init(dispatch, 4096);
    assert(vm_size(v) == 0);
    vm_add(v, 0);
    assert(vm_size(v) == 1);
    assert(vm_remove(v) == 0);
    assert(vm_size(v) == 0);
    vm_free(v);
}

void
test_ref_counts() {
    if (!(dispatch == rc_get_dispatch_table() ||
          dispatch == rcc_get_dispatch_table()))
        return;
    vm *v = vm_init(dispatch, 200);
    ptr bi = vm_boxed_int_init(v, 99);
    vm_add(v, bi);
    assert(P_GET_RC(bi) == 1);
    vm_remove(v);
    assert(vm_space_used(v) == 0);

    ptr w = vm_add(v, vm_wrapper_init(v, vm_boxed_int_init(v, 3)));
    assert(P_GET_RC(*SLOT_P(w, 0)) == 1);
    assert(vm_space_used(v) == NPTRS(4));
    vm_remove(v);
    vm_collect(v);
    assert(vm_space_used(v) == 0);

    vm_add(v, vm_array_init(v, 10, 0));
    assert(vm_space_used(v) == NPTRS(14));
    vm_remove(v);
    vm_collect(v);
    assert(vm_space_used(v) == 0);

    ptr a = vm_add(v, vm_array_init(v, 10, vm_boxed_int_init(v, 3)));
    assert(P_GET_RC(*SLOT_P(a, 3)) == 10);
    assert(vm_space_used(v) == NPTRS(16));
    vm_remove(v);
    vm_collect(v);
    assert(vm_space_used(v) == 0);

    vm_add(v, vm_wrapper_init(v, vm_boxed_int_init(v, 3)));
    assert(vm_space_used(v) == NPTRS(4));
    vm_set(v, 0, vm_wrapper_init(v, vm_boxed_int_init(v, 3)));
    vm_collect(v);
    assert(vm_space_used(v) == NPTRS(4));

    vm_free(v);
}

void
test_ref_count_colors() {
    if (dispatch != rcc_get_dispatch_table()) {
        return;
    }
    vm *v = vm_init(dispatch, 200);
    ptr bi = vm_boxed_int_init(v, 99);
    P_SET_COL(bi, COL_PURPLE);
    vm_add(v, bi);
    assert(P_GET_RC(bi) == 1);
    assert(P_GET_COL(bi) == COL_BLACK);
    vm_remove(v);
    assert(vm_space_used(v) == 0);

    vm_add(v, vm_boxed_int_init(v, 30));
    vm_add(v, vm_wrapper_init(v, vm_get(v, 0)));

    assert(P_GET_RC(vm_get(v, 0)) == 2);

    vm_remove(v);
    assert(P_GET_RC(vm_get(v, 0)) == 1);
    assert(P_GET_COL(vm_get(v, 0)) == COL_BLACK);

    vm_remove(v);
    vm_free(v);
}

void
test_collect() {
    vm *v = vm_init(dispatch, 4096);
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
    if (dispatch == rcc_get_dispatch_table() ||
        dispatch == rc_get_dispatch_table()) {
        assert(vm_space_used(v) == 0);
    }
    vm_free(v);
}

void
test_dump() {
    vm *v = vm_init(dispatch, 4096);
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
    vm *v = vm_init(dispatch, 9192);
    vm_add(v, vm_array_init(v, 20, vm_boxed_int_init(v, 20)));
    vm_collect(v);
    vm_add(v, vm_array_init(v, 40, 0));
    vm_collect(v);
    vm_free(v);
}

void
test_random_stuff() {
    vm *v = vm_init(dispatch, 32 * 1024);

    ptr bf = vm_boxed_float_init(v, 2.731);
    assert(bf);
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

    for (size_t n = 0; n < 400000; n++) {
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
    vm *v = vm_init(dispatch, 200 * 1024 * 1024);
    for (int i = 0; i < 100; i++) {
        vm_add(v, vm_array_init(v, 500, 0));
    }
    int n_roots = (int)vm_size(v);
    assert(n_roots == 100);
    for (int i = 0; i < 4000000; i++) {
        ptr rand_arr = vm_get(v, rand_n(n_roots));
        int n_els = (int)*SLOT_P(*SLOT_P(rand_arr, 0), 0);
        ptr p = random_object(v);
        vm_set_slot(v, rand_arr, 1 + rand_n(n_els), p);
    }
    vm_collect(v);
    vm_free(v);
}

void
test_collector(char *name, gc_dispatch *this_dispatch) {
    printf("== Running the %s Collector ==\n\n", name);
    dispatch = this_dispatch;
    PRINT_RUN(test_vm);
    PRINT_RUN(test_ref_counts);
    PRINT_RUN(test_ref_count_colors);
    PRINT_RUN(test_collect);
    PRINT_RUN(test_dump);
    PRINT_RUN(test_stack_overflow);
    PRINT_RUN(test_mark_stack_overflow);
    PRINT_RUN(test_random_stuff);
    PRINT_RUN(test_torture);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    gc_dispatch *dispatches[5] = {
        cg_get_dispatch_table(),
        rc_get_dispatch_table(),
        rcc_get_dispatch_table(),
        ms_get_dispatch_table(),
        cg_get_dispatch_table_optimized()
    };
    char *names[5] = {
        "Copying",
        "Reference Counting",
        "Cycle-collecting Reference Counting",
        "Mark & Sweep",
        "Optimized Copying"
    };
    for (size_t n = 0; n < 5; n++) {
        test_collector(names[n], dispatches[n]);
    }
    return 0;
}
