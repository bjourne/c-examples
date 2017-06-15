#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "datatypes/common.h"

void
test_shuffle() {
    int orig[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int new[10];
    memcpy(new, orig, sizeof(int) * 10);
    rand_shuffle(new, 10, sizeof(int));
    bool same = true;
    for (int i = 0; i < 10; i++) {
        same = same && (orig[i] == new[i]);
    }
    // The odds are low
    assert(!same);
    for (int i = 0; i < 10; i++) {
        printf("%d\n", new[i]);
    }
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_shuffle);
}
