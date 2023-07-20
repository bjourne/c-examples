// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
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
    uint32_t u32;
    d.v = 0;
    if (val < 0) {
        d.raw.sign = 1;
        u32 = (uint32_t)-val;
    } else {
        d.raw.sign = 0;
        u32 = (uint32_t)val;
    }
    if (!u32) {
        return 0;
    }

    // Index of most significant bit (MSB)
    uint8_t msb = 0;
    while ((1U << msb) <= (u32 >> 1)) {
        msb++;
    }

    // Mask msb.
    u32 -= (1 << msb);

    uint32_t sig;
    if (msb > 23) {
        // Index of the truncated part's MSB.
        int8_t trunc_msb = msb - 23;
        sig = u32 >> trunc_msb;

        // Upper bound of truncation range.
        uint32_t upper = 1 << trunc_msb;

        // Truncted value
        uint32_t trunc = u32 & (upper - 1);

        // Distance to the upper and lower bound (which is zero).
        uint32_t lo = trunc - 0;
        uint32_t hi = upper - trunc;

        // Round up if closer to upper bound than lower, or if
        // equally close round up if odd (so to even).
        if ((lo > hi) ||
            (lo == hi && (sig & 1))) {
            sig++;

            // Incrementing the sig may cause wrap-around in
            // which case we increase the msb.
            sig &= (1 << 23) - 1;
            msb += !sig;
        }
    } else {
        sig = u32 << (23 - msb);
    }
    uint8_t exp = msb + 127;

    d.raw.frac = sig;
    d.raw.exp = exp;
    return d.v;
}
