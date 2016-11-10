#ifndef DATATYPES_BITS_H
#define DATATYPES_BITS_H

#define BF_BASIC_MASK(n)            ((1 << (n)) - 1)
#define BF_MASK(start, len)         BF_BASIC_MASK(len) << (start)
#define BF_GET(x, start, len)       (((x) >> (start)) & BF_BASIC_MASK(len))
#define BF_SET(x, n, start, len)    ((x) &~ BF_MASK(start, len)) | ((n) << (start))

#endif
