#include <assert.h>
#include <string.h>
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

typedef struct {
    int msg_type;
    char desc[24];
} msg;

void
test_message_queue() {
    var_vector *v = var_vec_init(8, sizeof(msg));

    msg msg1 = (msg){3, "foo"};
    var_vec_add(v, &msg1);
    assert(v->used == 1);
    msg msg2;
    var_vec_remove(v, &msg2);
    assert(msg2.msg_type == 3);
    assert(!strcmp(msg2.desc, "foo"));
    assert(v->used == 0);
    var_vec_free(v);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_basic);
    PRINT_RUN(test_remove_at);
    PRINT_RUN(test_message_queue);
    return 0;
}
