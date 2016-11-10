#include "collectors/vm.h"

ptr
vm_remove(vm *me) {
    ptr p = v_remove(me->roots);
    ptr *ptr = &me->roots->array[me->roots->used];
    me->gc_dispatch->set_ptr(me->gc_obj, ptr, 0);
    return p;
}

vm *
vm_init(gc_dispatch *gc_dispatch, size_t size) {
    vm *me = malloc(sizeof(vm));
    me->roots = v_init(16);
    me->memory = (ptr)malloc(size);
    me->size = size;
    me->gc_dispatch = gc_dispatch;
    me->gc_obj = gc_dispatch->init(me->memory, size);
    return me;
}

void
vm_free(vm *me) {
    while (me->roots->used) {
        vm_remove(me);
    }
    v_free(me->roots);
    me->gc_dispatch->free(me->gc_obj);
    free((void*)me->memory);
    free(me);
}

ptr
vm_add(vm *me, ptr p) {
    v_add(me->roots, 0);
    ptr *ptr = &me->roots->array[me->roots->used - 1];
    me->gc_dispatch->set_new_ptr(me->gc_obj, ptr, p);
    return p;
}

ptr
vm_last(vm *me) {
    return v_peek(me->roots);
}

size_t
vm_size(vm *me) {
    return me->roots->used;
}

void
vm_set(vm *me, size_t i, ptr p) {
    if (i >= vm_size(me)) {
        error("Out of bounds %lu!", i);
    }
    me->gc_dispatch->set_ptr(me->gc_obj, &me->roots->array[i], p);
}

ptr
vm_get(vm *me, size_t i) {
    return me->roots->array[i];
}

void
vm_collect(vm *me) {
    me->gc_dispatch->collect(me->gc_obj, me->roots);
}

void
vm_set_slot(vm *me, ptr p_from, size_t i, ptr p) {
    me->gc_dispatch->set_ptr(me->gc_obj, SLOT_P(p_from, i), p);
}

static ptr
vm_allot(vm *me, size_t n_ptrs, int type) {
    size_t size = NPTRS(n_ptrs);
    void* gc_obj = me->gc_obj;
    gc_dispatch *gc_dispatch = me->gc_dispatch;
    if (!gc_dispatch->can_allot_p(gc_obj, size)) {
        vm_collect(me);
        if (!gc_dispatch->can_allot_p(gc_obj, size)) {
            error("Can't allocate %lu bytes! Space used %lu\n",
                  size, vm_space_used(me));
        }
    }
    return gc_dispatch->do_allot(gc_obj, type, size);
}


ptr
vm_boxed_int_init(vm *me, int value) {
    ptr item = vm_allot(me, 2, TYPE_INT);
    *SLOT_P(item, 0) = value;
    return item;
}

ptr
vm_boxed_float_init(vm *me, double value) {
    ptr item = vm_allot(me, 2, TYPE_FLOAT);
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
    me->gc_dispatch->set_new_ptr(me->gc_obj, SLOT_P(item, 0), vm_last(me));
    vm_remove(me);

    value = vm_last(me);
    ptr *base = SLOT_P(item, 1);
    for (size_t i = 0; i < n; i++) {
        me->gc_dispatch->set_new_ptr(me->gc_obj, base + i, value);
    }
    vm_remove(me);
    return item;
}

ptr
vm_wrapper_init(vm *me, ptr value) {
    vm_add(me, value);
    ptr item = vm_allot(me, 2, TYPE_WRAPPER);

    me->gc_dispatch->set_new_ptr(me->gc_obj, SLOT_P(item, 0), vm_last(me));
    vm_remove(me);
    return item;
}

void
vm_tree_dump(vm *me) {
    p_print_slots(0, me->roots->array, me->roots->used);
}

size_t
vm_space_used(vm *me) {
    return me->gc_dispatch->space_used(me->gc_obj);
}
