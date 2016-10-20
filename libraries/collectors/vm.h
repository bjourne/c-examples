#ifndef COLLECTORS_VM_H
#define COLLECTORS_VM_H

#include "datatypes/vector.h"
#include "collectors/common.h"
//#include "collectors/mark-sweep.h"
#include "collectors/copying.h"

typedef struct {
    copying_gc* mem_man;
    vector *roots;
} vm;

vm *vm_init(size_t max_used);
void vm_free(vm *v);

// Roots interface
ptr vm_add(vm *v, ptr p);
ptr vm_remove(vm *v);
ptr vm_last(vm *v);
size_t vm_size(vm *v);
void vm_set(vm *v, size_t i, ptr p);
ptr vm_get(vm *v, size_t i);

// Barriers & ref counting
void vm_set_slot(vm *v, ptr p_from, size_t i, ptr p);

// Object allocation
ptr vm_boxed_int_init(vm *v, int value);
ptr vm_boxed_float_init(vm *v, double value);
ptr vm_array_init(vm *v, size_t n, ptr value);
ptr vm_wrapper_init(vm *v, ptr value);

// Stats
void vm_tree_dump(vm *v);

#endif
