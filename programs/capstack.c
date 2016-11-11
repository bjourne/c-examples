// Demonstrates how to capture the process context in C. This is
// required when implementing tracing gc.
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <ucontext.h>
#include "collectors/mark-sweep.h"
#include "datatypes/common.h"
#include "datatypes/vector.h"

static ptr
get_stack_pointer() {
    register ptr stack asm("rsp");
    return stack + 16;
}

typedef struct {
    ptr start;
    size_t size;
    ptr stack_base;
} root_finder;

static root_finder *
rf_init(ptr start, size_t size, ptr stack_base) {
    root_finder *finder = (root_finder *)malloc(sizeof(root_finder));
    finder->start = start;
    finder->size = size;
    finder->stack_base = stack_base;
    return finder;
}

static void
rf_free(root_finder *finder) {
    free(finder);
}

void
rf_collect_stack_roots(root_finder *me, vector *v) {
    // By pushing all registers onto the stack we don't have to
    // inspect them separately.
    asm volatile("push %rax\n\t"
                 "push %rbx\n\t"
                 "push %rcx\n\t"
                 "push %rdx\n\t"
                 "push %rdi\n\t"
                 "push %rsi\n\t"
                 "push %rbp\n\t"
                 "push %r8\n\t"
                 "push %r9\n\t"
                 "push %r10\n\t"
                 "push %r11\n\t"
                 "push %r12\n\t"
                 "push %r13\n\t"
                 "push %r14\n\t"
                 "push %r15\n\t");

    ptr iter = get_stack_pointer();
    ptr end = me->stack_base;
    int idx = 0;
    ptr mem_start = me->start;
    ptr mem_end = me->start + me->size;
    while (iter < end) {
        ptr p = AT(iter);
        if (p >= mem_start && p < mem_end) {
            v_add(v, p);
        }
        iter += sizeof(ptr);
        idx++;
    }

    asm volatile("pop %r15\n\t"
                 "pop %r14\n\t"
                 "pop %r13\n\t"
                 "pop %r12\n\t"
                 "pop %r11\n\t"
                 "pop %r10\n\t"
                 "pop %r9\n\t"
                 "pop %r8\n\t"
                 "pop %rbp\n\t"
                 "pop %rsi\n\t"
                 "pop %rdi\n\t"
                 "pop %rdx\n\t"
                 "pop %rcx\n\t"
                 "pop %rbx\n\t"
                 "pop %rax\n\t");
}

typedef struct {
    root_finder *rf;
    mark_sweep_gc *ms;
    ptr start;
    size_t size;
    vector *roots;
} c_allocator;

c_allocator *
ca_init(ptr stack_base, size_t size) {
    c_allocator *me = (c_allocator *)malloc(sizeof(c_allocator));
    me->size = size;
    me->start = (ptr)malloc(me->size);
    me->rf = rf_init(me->start, me->size, stack_base + 0);
    me->ms = ms_init(me->start, me->size);
    me->roots = v_init(16);
    return me;
}

void
ca_free(c_allocator *me) {
    rf_free(me->rf);
    ms_free(me->ms);
    v_free(me->roots);
    free((void *)me->start);
    free(me);
}

void
ca_collect(c_allocator *me) {
    me->roots->used = 0;
    rf_collect_stack_roots(me->rf, me->roots);
    ms_collect(me->ms, me->roots);
}

ptr
ca_allot(c_allocator *me, size_t size) {
    if (!ms_can_allot_p(me->ms, size)) {
        ca_collect(me);
        if (!ms_can_allot_p(me->ms, size)) {
            error("Can't allocate %lu bytes! Space used %lu\n",
                  size, ms_space_used(me->ms));
        }
    }
    return ms_do_allot(me->ms, TYPE_INT, size);
}

static c_allocator *
global_ca = NULL;

void
f2(ptr p) {
    for (int n = 0; n < 5000; n++) {
        ca_allot(global_ca, rand_n(1000) + 100);
    }
}

void
f1() {
    ptr p = ca_allot(global_ca, 1024);
    f2(ca_allot(global_ca, 1000));
    // It works! p is still a working pointer!
    assert(QF_GET_BLOCK_SIZE(p) == 1024);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    global_ca = ca_init(get_stack_pointer(), 65536);
    f1();
    ca_free(global_ca);
    return 0;
}
