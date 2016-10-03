#include <assert.h>
#include "../vector.h"

int main(int argc, char *argv[]) {
    vector *v = v_init(10);

    for (int i = 0; i < 11; i++) {
        v_add(v, i);
    }

    assert(v->size == 15);

    v_free(v);
    return 0;
}
