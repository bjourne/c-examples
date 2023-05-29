#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"

void
test_malloc_aligned() {
    int dim = 1 << 15;
    int n_bytes = dim * dim * 4;
    char *data = malloc_aligned(64, 1 << 15);
    assert(data);
    free(data);
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_malloc_aligned);
}
