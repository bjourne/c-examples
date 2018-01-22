#ifndef DATATYPES_BITS_H
#define DATATYPES_BITS_H

#include "datatypes/common.h"

// Bit munching
#define ALIGN(a, b) ((a + (b - 1)) & ~(b - 1))

// Note 1L, to make it work on 64bit types.
#define BF_LIT_BITS(n)              ((1L << (n)) - 1)
#define BF_MASK(start, len)         (BF_LIT_BITS(len) << (start))
#define BF_GET(x, start, len)       (((x) >> (start)) & BF_LIT_BITS(len))
#define BF_MERGE(x, n, start, len)  (((x) & ~BF_MASK(start, len)) | ((n) << (start)))

#define P_GET(p, start, len)        BF_GET(AT(p), start, len)
#define P_SET(p, n, start, len)     AT(p) = BF_MERGE(AT(p), n, start, len)

// This function is borrowed from math.h in musl.
// BW = BitWise
inline unsigned int
BW_FLOAT_TO_UINT(float f) {
	union {
            float f;
            unsigned int u;
        } u;
	u.f = f;
	return u.u;
}

inline float
BW_PTR_TO_FLOAT(ptr p) {
    union {
        float f;
        ptr p;
    } u;
    u.p = p;
    return u.f;
}

inline unsigned int
BIT_COUNT(ptr p) {
    #ifdef _MSC_VER
    // How does this work?
    uint64_t k1 = 0x5555555555555555ll;
    uint64_t k2 = 0x3333333333333333ll;
    uint64_t k4 = 0x0f0f0f0f0f0f0f0fll;
    uint64_t kf = 0x0101010101010101ll;
    ptr ks = 56;
    p = p - ((p >> 1) & k1);
    p = (p & k2) + ((p >> 2) & k2);
    p = (p + (p >> 4)) & k4;
    p = (p * kf) >> ks;
    return (unsigned int)p;
    #else
    return __builtin_popcountl(p);
    #endif
}

#endif
