#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include "datatypes/heap.h"

int main(int argc, char *argv[]) {

    vector *v = v_init(5);

    for (int i = 0; i < 10; i++) {
        hp_add(v, 100 - i);
    }

    while (v->used) {
        printf("el %" PRIu64 "\n", hp_remove(v));
    }
    v_free(v);
    return 0;
}
