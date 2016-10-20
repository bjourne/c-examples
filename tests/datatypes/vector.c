#include <assert.h>
#include "datatypes/vector.h"

void
test_basic() {
    vector *v = v_init(10);

    for (int i = 0; i < 11; i++) {
        v_add(v, i);
    }

    assert(v->size == 15);

    v_free(v);
}

void
test_remove_at() {
    vector *v = v_init(32);

    for (int i = 0; i < 10; i++) {
        v_add(v, i);
    }
    assert(v_remove_at(v, 4) == 4);
    assert(v->used == 9);
    assert(v->array[4] == 5);
    assert(v_remove_at(v, 8) == 9);
    assert(v->used == 8);
    v_free(v);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_basic);
    PRINT_RUN(test_remove_at);
    return 0;
}
