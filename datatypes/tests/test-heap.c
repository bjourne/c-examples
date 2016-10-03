#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "../heap.h"

/* size_t rand_n(size_t n) { */
/*     return rand() % n; */
/* } */

int main(int argc, char *argv[]) {

    vector *v = v_init(5);

    for (int i = 0; i < 10; i++) {
        hp_insert(v, 100 - i);
    }

    while (v->used) {
        printf("el %lu\n", hp_pop(v));
    }

    /* hp_insert(v, 10); */
    /* hp_insert(v, 20); */
    /* hp_insert(v, 5); */
    /* hp_insert(v, 99); */

    /* hp_insert(v, 21); */

    /* for (int i = 0; i < v->used; i++) { */
    /*     printf("%d = %lu\n", i, v->array[i]); */
    /* } */

    v_free(v);
    return 0;
}
