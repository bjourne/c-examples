#include <assert.h>
#include "collectors/vm.h"
#include "collectors/mark-sweep.h"

void
test_collect_1() {
    vm *v = vm_init(ms_get_dispatch_table(), 4096);

    mark_sweep_gc *ms = (mark_sweep_gc *)v->mem_man;
    assert(ms->qf->n_blocks == 1);
    assert(QF_GET_BLOCK_SIZE(ms->start) == 4096);

    ptr p = vm_add(v, vm_boxed_int_init(v, 20));
    ptr rel_block_addr = p - ms->start;
    assert(rel_block_addr == QF_LARGE_BLOCK_SIZE(16) - 16);

    ms_collect(ms, v->roots);
    assert(ms->used == 16);
    assert(ms->qf->free_space == 4080);
    vm_free(v);
}

void
test_collect_2() {
    mark_sweep_gc *ms = ms_init(4096);

    vector *roots = v_init(16);
    ptr p = ms_do_allot(ms, 4080);
    assert(p);
    assert(ms->qf->n_blocks == 1);


    v_add(roots, p);
    P_SET_TYPE(p, TYPE_INT);

    ms_collect(ms, roots);

    // Blocks must be unmarked
    assert(!P_GET_MARK(p));
    assert(ms->used == 16);

    v_free(roots);
    ms_free(ms);
}

void
test_collect_3() {
    mark_sweep_gc *ms = ms_init(4096);
    vector *roots = v_init(16);

    ptr p = ms_do_allot(ms, 16);
    P_SET_TYPE(p, TYPE_INT);
    ptr rel_addr = p - ms->start;
    assert(rel_addr == 1008);
    v_add(roots, p);

    p = ms_do_allot(ms, 176);
    P_SET_TYPE(p, TYPE_INT);
    v_add(roots, p);
    assert(ms->used == 176 + 16);

    ms_collect(ms, roots);

    p = ms_do_allot(ms, 16);
    v_add(roots, p);

    qf_print(ms->qf, ms->start, ms->size);

    assert(!ms_can_allot_p(ms, 336));
    v_free(roots);
    ms_free(ms);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_collect_1);
    PRINT_RUN(test_collect_2);
    PRINT_RUN(test_collect_3);
}
