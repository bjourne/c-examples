// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdio.h>
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


uint32_t
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

uint32_t
ieee754_i32_to_f32(uint32_t val) {
    ieee754_f32 d;
    if (val < 0) {
        d.raw.sign = 1;
        val = -val;
    } else {
        d.raw.sign = 0;
    }
    if (!val) {
        return 0;
    }
    return d.v;
}
