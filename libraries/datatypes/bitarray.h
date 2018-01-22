#ifndef DATATYPES_BITARRAY_H
#define DATATYPES_BITARRAY_H

#include <stdbool.h>
#include "datatypes/common.h"

// How this works, I don't know!
inline ptr
bw_log2(ptr x) {
    ptr n;
#if defined(CPU_32)
    #if defined(_MSC_VER)
    _BitScanReverse((unsigned long*)&n, x);
    #else
    asm("bsr %1, %0;" : "=r"(n) : "r"(x));
    #endif

#elif defined(CPU_64)

    #if defined(_MSC_VER)
    n = 0;
    _BitScanReverse64((unsigned long*)&n, x);
    #else
    asm("bsr %1, %0;" : "=r"(n) : "r"(x));
    #endif
#endif
  return n;
}

inline int
rightmost_clear_bit(ptr x) {
    return (int)bw_log2(~x & (x + 1));
}

inline int
rightmost_set_bit(ptr x) {
    return (int)bw_log2(x & (~x + 1));
}

#define BA_WORD_BITS   (8 * sizeof(ptr))

#define BA_EACH_UNSET_RANGE(ba, body)                                   \
    for (int _iter = ba_next_unset_bit(ba, 0); _iter < ba->n_bits; ) {  \
        int _next = ba_next_set_bit(ba, _iter);                         \
        int addr = _iter;                                               \
        int size = _next - _iter;                                       \
        { body }                                                        \
        _iter = ba_next_unset_bit(ba, _next);                           \
    }


typedef struct {
    ptr bits;
    int n_bits;
    int n_words;
} bitarray;

bitarray *ba_init(int n_bits);
void ba_free(bitarray *me);

void ba_set_bit(bitarray *me, int addr);
bool ba_get_bit(bitarray *me, int addr);
void ba_set_bit_range(bitarray *me, int addr, int n);
void ba_clear(bitarray *me);

int ba_next_unset_bit(bitarray *me, int start);
int ba_next_set_bit(bitarray *me, int start);

// Count of the number of lit bits in a bitarray.
unsigned int ba_bitsum(bitarray *me);

#endif
