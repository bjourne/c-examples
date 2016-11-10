#include <assert.h>
#include "collectors/vm.h"
#include "collectors/mark-sweep.h"

void
test_collect_1() {
    vm *v = vm_init(ms_get_dispatch_table(), 4096);

    mark_sweep_gc *ms = (mark_sweep_gc *)v->gc_obj;
    assert(ms->qf->n_blocks == 1);
    assert(QF_GET_BLOCK_SIZE(ms->qf->start) == 4096);

    ptr p = vm_add(v, vm_boxed_int_init(v, 20));
    ptr rel_block_addr = p - ms->qf->start;
    assert(rel_block_addr == QF_LARGE_BLOCK_SIZE(16) - 16);

    ms_collect(ms, v->roots);
    assert(ms_space_used(ms) == 16);
    assert(ms->qf->free_space == 4080);
    vm_free(v);
}

void
test_collect_2() {
    ptr mem = (ptr)malloc(4096);
    mark_sweep_gc *ms = ms_init(mem, 4096);

    vector *roots = v_init(16);
    ptr p = ms_do_allot(ms, TYPE_INT, 4080);
    assert(ms_space_used(ms) == 4080);
    assert(p);
    assert(ms->qf->n_blocks == 1);
    v_add(roots, p);

    ms_collect(ms, roots);

    // Blocks must be unmarked
    assert(!P_GET_MARK(p));
    assert(ms_space_used(ms) == 4080);

    v_free(roots);
    ms_free(ms);
    free((void*)mem);
}

void
test_collect_3() {
    ptr mem = (ptr)malloc(4096);
    mark_sweep_gc *ms = ms_init(mem, 4096);
    vector *roots = v_init(16);

    ptr p = ms_do_allot(ms, TYPE_INT, 16);
    ptr rel_addr = p - ms->qf->start;
    assert(rel_addr == 1008);
    v_add(roots, p);

    p = ms_do_allot(ms, TYPE_INT, 176);
    v_add(roots, p);
    assert(ms_space_used(ms) == 176 + 16);

    ms_collect(ms, roots);

    p = ms_do_allot(ms, TYPE_INT, 16);
    v_add(roots, p);

    qf_print(ms->qf);

    assert(!ms_can_allot_p(ms, 336));

    v_free(roots);
    ms_free(ms);
    free((void*)mem);
}

void
test_do_allot() {
    ptr mem = (ptr)malloc(4096);
    mark_sweep_gc *ms = ms_init(mem, 4096);

    ms_do_allot(ms, TYPE_INT, 176);
    assert(QF_GET_BLOCK_SIZE(mem) == 176);

    ms_free(ms);
    free((void*)mem);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_collect_1);
    PRINT_RUN(test_collect_2);
    PRINT_RUN(test_collect_3);
    PRINT_RUN(test_do_allot);
}
