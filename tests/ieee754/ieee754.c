// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/bits.h"
#include "datatypes/common.h"
#include "ieee754/ieee754.h"

static uint32_t log2ceil(uint32_t n) {
    uint32_t q = n >> 1;
    uint32_t r = 0;
    while (q) {
        q = q >> 1;
        r++;
    }
    return r;
}

static void
test_i32_to_f32() {
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
test_f32_to_i32() {
    assert(ieee754_i32_to_f32(0) == 0);

    printf("%d\n", log2ceil(3));
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_i32_to_f32);
    PRINT_RUN(test_f32_to_i32);
}
