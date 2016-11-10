#ifndef COLLECTORS_VM_H
#define COLLECTORS_VM_H

#include "datatypes/vector.h"
#include "collectors/common.h"

typedef struct {
    vector *roots;
    ptr memory;
    size_t size;
    gc_dispatch *gc_dispatch;
    void* gc_obj;
} vm;

vm *vm_init(gc_dispatch *gc_dispatch, size_t max_used);
void vm_free(vm *me);

// Roots interface
ptr vm_add(vm *me, ptr p);
ptr vm_remove(vm *me);
ptr vm_last(vm *me);
size_t vm_size(vm *me);
void vm_set(vm *me, size_t i, ptr p);
ptr vm_get(vm *me, size_t i);

// Force collection
void vm_collect(vm *me);

// Barriers & ref counting
void vm_set_slot(vm *me, ptr p_from, size_t i, ptr p);

// Object allocation
ptr vm_boxed_int_init(vm *me, int value);
ptr vm_boxed_float_init(vm *me, double value);
ptr vm_array_init(vm *me, int n, ptr value);
ptr vm_wrapper_init(vm *me, ptr value);

// Stats
void vm_tree_dump(vm *me);
size_t vm_space_used(vm *me);

#endif
