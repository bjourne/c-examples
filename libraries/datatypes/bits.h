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

#endif
