// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"
#include "ieee754/ieee754.h"

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
        5,
        -5,
        33,
        -33,
        999999,
        9999999,
        99999999,
    };
    float floats[] = {
        5.0,
        -5.0,
        33.0,
        -33.0,
        999999.0f,
        9999999.0f,
        99999999.0f
    };
    printf("really1 %.2f\n", 99999999.0f);
    printf("really2 %.2f\n", floats[6]);
    for (int i = 0; i < ARRAY_SIZE(floats); i++) {
        float f = floats[i];
        int v = ints[i];
        printf("%10d -> %18.6f\n", v, f);

        uint32_t exp_f32 = BW_FLOAT_TO_UINT(f);
        uint32_t got_f32 = ieee754_i32_to_f32(v);

        if (exp_f32 != got_f32) {
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
    PRINT_RUN(test_i32_to_f32);
    PRINT_RUN(test_f32_to_i32);
}
