// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"
#include "ieee754/ieee754.h"
#include "random/random.h"

static void
test_f32_to_i32() {
    float floats[] = {
        25.0,
        -25.0, -123456.0,
        123456789.0, 0.0,
        0.01,
        -123.99
    };
    int ints[] = {
        25,
        -25, -123456,
        123456792, 0,
        0,
        -123
    };
    for (int i = 0; i < ARRAY_SIZE(floats); i++) {
        float f = floats[i];
        int v = ints[i];
        printf("%18.4f -> %10d\n", f, v);
        assert(ieee754_f32_to_i32(BW_FLOAT_TO_UINT(f)) == v);
    }
}

static void
test_i32_to_f32() {
    int ints[] = {
        1,
        5,
        -5,
        33,
        -33,
        999999,
        9999999,
        99999999,
        -1234,
        (1 << 23) + 20,
        (1 << 24) + 1,
        (1 << 24) + 2,
        (1 << 24) + 3,
        (1 << 24) + 4,
        (1 << 24) + 5,
        (1 << 24) + 6,
        (1 << 24) + 7,
        (1 << 24) + 8,
        (1 << 24) + 9,
        (1 << 24) + 10,
        2147483600
    };
    for (size_t i = 0; i < ARRAY_SIZE(ints); i++) {
        int v = ints[i];

        float f = (float)v;
        uint32_t exp_f32 = BW_FLOAT_TO_UINT(f);
        uint32_t got_f32 = ieee754_i32_to_f32(v);

        /* ieee754_print_bits(got_f32); */
        /* printf("\n"); */

        if (exp_f32 != got_f32) {
            printf("%d failed:\n", v);
            printf("expected ");
            ieee754_print_bits(exp_f32);
            printf(" (%18.8f)\n", f);
            printf("got      ");
            ieee754_print_bits(got_f32);
            printf(" (%18.8f)\n", BW_UINT_TO_FLOAT(got_f32));
            printf("\n");
        }
        assert(exp_f32 == got_f32);
    }
}

void
test_i32_to_f32_random() {
    for (int i = 0; i < 10000000; i++) {
        int32_t v = rnd_pcg32_rand();
        float f = (float)v;
        uint32_t exp_f32 = BW_FLOAT_TO_UINT(f);
        uint32_t got_f32 = ieee754_i32_to_f32(v);
        if (exp_f32 != got_f32) {
            printf("%d failed:\n", v);
            printf("expected ");
            ieee754_print_bits(exp_f32);
            printf(" (%18.8f)\n", f);
            printf("got      ");
            ieee754_print_bits(got_f32);
            printf(" (%18.8f)\n", BW_UINT_TO_FLOAT(got_f32));
            printf("\n");
        }
        assert(exp_f32 == got_f32);
    }
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_f32_to_i32);
    PRINT_RUN(test_i32_to_f32);
    PRINT_RUN(test_i32_to_f32_random);
}
