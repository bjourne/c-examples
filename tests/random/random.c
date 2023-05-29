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
    uint32_t lim = rnd_pcg32_rand_range(50);
    printf("Numbers in range 0 <= r < %u:\n", lim);
    for (int i = 0; i < 10; i++) {
        printf("%5u\n", rnd_pcg32_rand_range(lim));
    }
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_random);
    PRINT_RUN(test_rand_range);
}
