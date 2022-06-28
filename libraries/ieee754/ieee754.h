// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>

#ifndef IEEE754_H
#define IEEE754_H

#include <stdint.h>

// Since we are bit-twiddling we only use the uint32_t and uint64_t
// types in the interface.
uint32_t ieee754_f32_to_i32(uint32_t flt);
uint32_t ieee754_i32_to_f32(uint32_t val);


#endif
