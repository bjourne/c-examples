// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "collectors/mark-sweep-bits.h"

void
test_bit_markings() {
    size_t heap_size = 1024 * 1024;
    ptr mem = (ptr)malloc(heap_size);
    mark_sweep_bits_gc *ms = msb_init(mem, heap_size);

    assert(ms->ba->n_words == (heap_size / 16 / BA_WORD_BITS));

    vector *roots = v_init(16);
    v_add(roots, msb_do_allot(ms, TYPE_INT, 1000));

    msb_collect(ms, roots);

    msb_free(ms);
    free((void*)mem);
    v_free(roots);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_bit_markings);
}
