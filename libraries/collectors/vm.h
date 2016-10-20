#ifndef COLLECTORS_VM_H
#define COLLECTORS_VM_H

#include "collectors/common.h"
#include "collectors/mark-sweep.h"

typedef struct {
    mark_sweep_gc* mem_man;
    vector *roots;
} vm;

vm *vm_init(size_t max_used);
void vm_free(vm *v);

// Roots interface
ptr vm_add(vm *v, ptr p);
ptr vm_remove(vm *v);
ptr vm_last(vm *v);

// Object allocation
ptr vm_boxed_int_init(vm *v, int value);
ptr vm_boxed_float_init(vm *v, double value);
ptr vm_array_init(vm *v, size_t n, ptr value);


#endif
