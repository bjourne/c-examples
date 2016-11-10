#include "collectors/vm.h"

ptr
vm_remove(vm *me) {
    ptr p = v_remove(me->roots);
    ptr *ptr = &me->roots->array[me->roots->used];
    me->dispatch->set_ptr(me->mem_man, ptr, 0);
    return p;
}

vm *
vm_init(gc_dispatch *dispatch, size_t size) {
    vm *me = malloc(sizeof(vm));
    me->dispatch = dispatch;
    me->roots = v_init(16);
    me->mem_man = dispatch->init(size);
    return me;
}

void
vm_free(vm *v) {
    while (v->roots->used) {
        vm_remove(v);
    }
    v_free(v->roots);
    v->dispatch->free(v->mem_man);
    free(v);
}

ptr
vm_add(vm *v, ptr p) {
    v_add(v->roots, 0);
    ptr *ptr = &v->roots->array[v->roots->used - 1];
    v->dispatch->set_new_ptr(v->mem_man, ptr, p);
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
vm_set(vm *me, size_t i, ptr p) {
    if (i >= vm_size(me)) {
        error("Out of bounds %lu!", i);
    }
    me->dispatch->set_ptr(me->mem_man, &me->roots->array[i], p);
}

ptr
vm_get(vm *v, size_t i) {
    return v->roots->array[i];
}

void
vm_collect(vm *me) {
    me->dispatch->collect(me->mem_man, me->roots);
}

void
vm_set_slot(vm *me, ptr p_from, size_t i, ptr p) {
    me->dispatch->set_ptr(me->mem_man, SLOT_P(p_from, i), p);
}

static ptr
vm_allot(vm *me, size_t n_ptrs, int type) {
    size_t size = NPTRS(n_ptrs);
    void* ms = me->mem_man;
    if (!me->dispatch->can_allot_p(ms, size)) {
        vm_collect(me);
        if (!me->dispatch->can_allot_p(ms, size)) {
            error("Can't allocate %lu bytes! Space used %lu\n",
                  size, vm_space_used(me));
        }
    }
    return me->dispatch->do_allot(ms, type, size);
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

// Pointers are pushed onto the root stack because we don't want the
// collectors to sweep those objects while we are using them. The
// copying collector might change object addresses and the reference
// counter might free objects without any references.
ptr
vm_array_init(vm *me, int n, ptr value) {
    vm_add(me, value);

    vm_add(me, vm_boxed_int_init(me, n));
    ptr item = vm_allot(me, 2 + n, TYPE_ARRAY);
    me->dispatch->set_new_ptr(me->mem_man, SLOT_P(item, 0), vm_last(me));
    vm_remove(me);

    value = vm_last(me);
    ptr *base = SLOT_P(item, 1);
    for (size_t i = 0; i < n; i++) {
        me->dispatch->set_new_ptr(me->mem_man, base + i, value);
    }
    vm_remove(me);
    return item;
}

ptr
vm_wrapper_init(vm *me, ptr value) {
    vm_add(me, value);
    ptr item = vm_allot(me, 2, TYPE_WRAPPER);

    me->dispatch->set_new_ptr(me->mem_man, SLOT_P(item, 0), vm_last(me));
    vm_remove(me);
    return item;
}

void
vm_tree_dump(vm *me) {
    p_print_slots(0, me->roots->array, me->roots->used);
}

size_t
vm_space_used(vm *me) {
    return me->dispatch->space_used(me->mem_man);
}
