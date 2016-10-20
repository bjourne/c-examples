#include "collectors/vm.h"

ptr
vm_remove(vm *v) {
    ptr p = v_remove(v->roots);
    ptr *ptr = &v->roots->array[v->roots->used];
    cg_set_ptr(v->mem_man, ptr, 0);
    return p;
}

vm *
vm_init(size_t max_used) {
    vm *v = malloc(sizeof(vm));
    v->roots = v_init(16);
    v->mem_man = cg_init(max_used);
    return v;
}

void
vm_free(vm *v) {
    while (v->roots->used) {
        vm_remove(v);
    }
    v_free(v->roots);
    cg_free(v->mem_man);
    free(v);
}

ptr
vm_add(vm *v, ptr p) {
    v_add(v->roots, 0);
    ptr *ptr = &v->roots->array[v->roots->used - 1];
    cg_set_new_ptr(v->mem_man, ptr, p);
    return p;
}

ptr
vm_last(vm *v) {
    return v_peek(v->roots);
}

size_t
vm_size(vm *v) {
    return v->roots->used;
}

void
vm_set(vm *v, size_t i, ptr p) {
    if (i >= vm_size(v)) {
        error("Out of bounds %lu!", i);
    }
    cg_set_ptr(v->mem_man, &v->roots->array[i], p);
}

ptr
vm_get(vm *v, size_t i) {
    return v->roots->array[i];
}

void
vm_set_slot(vm *v, ptr p_from, size_t i, ptr p) {
    cg_set_ptr(v->mem_man, SLOT_P(p_from, i), p);
}

ptr
vm_allot(vm *v, size_t n_ptrs, uint type) {
    size_t n_bytes = NPTRS(n_ptrs);
    copying_gc* ms = v->mem_man;
    if (!cg_can_allot_p(ms, n_bytes)) {
        cg_collect(ms, v->roots);
        if (!cg_can_allot_p(ms, n_bytes)) {
            error("Can't allocate %lu bytes!\n", n_bytes);
        }
    }
    return cg_do_allot(ms, n_bytes, type);
}


ptr
vm_boxed_int_init(vm *v, int value) {
    ptr item = vm_allot(v, 2, TYPE_INT);
    *SLOT_P(item, 0) = value;
    return item;
}

ptr
vm_boxed_float_init(vm *v, double value) {
    ptr item = vm_allot(v, 2, TYPE_FLOAT);
    *(double *)SLOT_P(item, 0) = value;
    return item;
}

/* Pointers are pushed onto the root stack because we don't want the
   collectors to sweep those objects while we are using them. The
   copying collector might change object addresses and the reference
   counter might free objects without any references. */
ptr
vm_array_init(vm *v, size_t n, ptr value) {
    vm_add(v, value);

    vm_add(v, vm_boxed_int_init(v, n));
    ptr item = vm_allot(v, 2 + n, TYPE_ARRAY);
    cg_set_new_ptr(v->mem_man, SLOT_P(item, 0), vm_last(v));
    vm_remove(v);

    value = vm_last(v);
    ptr *base = SLOT_P(item, 1);
    for (size_t i = 0; i < n; i++) {
        cg_set_new_ptr(v->mem_man, base + i, value);
    }
    vm_remove(v);
    return item;
}

ptr
vm_wrapper_init(vm *v, ptr value) {
    vm_add(v, value);
    ptr item = vm_allot(v, 2, TYPE_WRAPPER);

    cg_set_new_ptr(v->mem_man, SLOT_P(item, 0), vm_last(v));
    vm_remove(v);
    return item;
}

void
vm_tree_dump(vm *v) {
    p_print_slots(0, v->roots->array, v->roots->used);
}
