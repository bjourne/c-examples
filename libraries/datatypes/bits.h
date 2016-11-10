#ifndef DATATYPES_BITS_H
#define DATATYPES_BITS_H

#define BF_LIT_BITS(n)              ((1 << (n)) - 1)
#define BF_MASK(start, len)         (BF_LIT_BITS(len) << (start))
#define BF_GET(x, start, len)       (((x) >> (start)) & BF_LIT_BITS(len))
#define BF_SET(x, n, start, len)    (((x) & ~BF_MASK(start, len)) | ((n) << (start)))

// Bit munching
#define ALIGN(a, b) ((a + (b - 1)) & ~(b - 1))

#endif
