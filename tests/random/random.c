// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <time.h>
#include "datatypes/common.h"
#include "random/random.h"

void
test_random() {
    printf("Random 32-bit numbers:\n");
    for (int i = 0; i < 10; i++) {
        printf("%12u\n", rnd_pcg32_rand());
    }
}

void
test_rand_range() {
    uint32_t lim = 1 + rnd_pcg32_rand_range(49);
    printf("Numbers in range 0 <= r < %u:\n", lim);
    for (int i = 0; i < 10; i++) {
        int r = rnd_pcg32_rand_range(lim);
        assert(r >= 0 && r < lim);
        printf("%5u\n", r);
    }
}

void
test_initing() {
    // Setting the seed initializes the library
    rnd_pcg32 bef = rnd_pcg32_get_state();
    assert(!bef.initialized);
    rnd_pcg32_seed(1007, 33);
    rnd_pcg32 aft = rnd_pcg32_get_state();
    assert(aft.initialized);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_initing);
    PRINT_RUN(test_random);
    PRINT_RUN(test_rand_range);

}
