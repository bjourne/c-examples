// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include "datatypes/common.h"
#include "ieee754.h"

typedef struct {
    uint32_t frac : 23;
    uint32_t exp : 8;
    uint32_t sign : 1;
} ieee754_sp;

typedef union {
    uint32_t v;
    ieee754_sp raw;
} ieee754_f32;

int32_t
ieee754_f32_to_i32(uint32_t flt) {

    ieee754_f32 d;
    d.v = flt;
    int32_t exp = d.raw.exp;
    if (!exp)
        return 0;
    exp -= 127;
    uint32_t ret = (1 << 23) | d.raw.frac;
    if (exp < 23)
        ret = ret >> (23 - exp);
    else if (exp > 23)
        ret = ret << (exp - 23);
    return d.raw.sign ? -ret : ret;
}

static uint32_t
log2floor(uint32_t n) {
    uint32_t q = n >> 1;
    uint32_t r = 0;
    while (q) {
        q = q >> 1;
        r++;
    }
    return r;
}

void
ieee754_print_bits(uint32_t f) {
    putchar(f & (1 << 31) ? '1' : '0');
    printf("  ");
    for (int i = 30; i >= 27; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    putchar('_');
    for (int i = 26; i >= 23; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    printf("  ");
    for (int i = 22; i >= 19; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    putchar('_');
    for (int i = 18; i >= 15; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    putchar('_');
    for (int i = 14; i >= 11; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    putchar('_');
    for (int i = 10; i >= 7; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    putchar('_');
    for (int i = 6; i >= 3; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
    putchar('_');
    for (int i = 2; i >= 0; i--) {
        putchar(f & (1 << i) ? '1' : '0');
    }
}

uint32_t
ieee754_i32_to_f32(int32_t val) {
    ieee754_f32 d;
    d.v = 0;
    if (val < 0) {
        d.raw.sign = 1;
        val = -val;
    } else {
        d.raw.sign = 0;
    }

    if (!val) {
        return 0;
    }
    int32_t exp = log2floor(val);

    // Subtract MSB
    uint32_t mask = (1 << exp);
    val = val - mask;
    mask >>= 1;

    for (int i = 22; i >= MAX(22 - exp + 1, 0); i--) {
        if (val >= mask) {
            val -= mask;
            d.raw.frac |= (1 << i);
        }
        mask >>= 1;
    }
    d.raw.exp = exp + 127;
    return d.v;
}
