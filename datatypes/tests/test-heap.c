#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "../heap.h"

int main(int argc, char *argv[]) {

    vector *v = v_init(5);

    for (int i = 0; i < 10; i++) {
        hp_add(v, 100 - i);
    }

    while (v->used) {
        printf("el %lu\n", hp_pop(v));
    }
    v_free(v);
    return 0;
}
