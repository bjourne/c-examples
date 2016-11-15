#include <string.h>
#include "bitarray.h"

#define WORD_BITS   (8 * sizeof(ptr))
#define WORD_MASK   (WORD_BITS - 1)

#define WORD_IDX(i)     (i / WORD_BITS)

void
ba_clear(bitarray *me) {
    memset((void *)me->bits, 0, me->n_words * sizeof(ptr));
}

bitarray *
ba_init(int n_bits) {
    bitarray *me = (bitarray *)malloc(sizeof(bitarray));
    me->n_bits = n_bits;
    me->n_words = (n_bits + WORD_BITS - 1) / WORD_BITS;
    me->bits = (ptr)malloc(me->n_words * sizeof(ptr));
    ba_clear(me);
    return me;
}

bool
ba_get_bit(bitarray *me, int addr) {
    int word_idx = addr / WORD_BITS;
    int bit_idx = addr & WORD_MASK;
    ptr p = me->bits + word_idx * sizeof(ptr);
    return AT(p) & (1L << bit_idx);
}

void
ba_set_bit(bitarray *me, int addr) {
    int word_idx = addr / WORD_BITS;
    int bit_idx = addr & WORD_MASK;
    ptr p = me->bits + word_idx * sizeof(ptr);
    AT(p) |= 1L << bit_idx;
}

void
ba_set_bit_range(bitarray *me, int addr, int n) {
    int word_start_idx = addr / WORD_BITS;
    int bit_start_idx = addr & WORD_MASK;
    int word_end_idx = (addr + n) / WORD_BITS;
    int bit_end_idx = (addr + n) & WORD_MASK;

    ptr start_mask = (1L << bit_start_idx) - 1L;
    ptr end_mask = (1L << bit_end_idx) - 1L;

    ptr p = me->bits + word_start_idx * sizeof(ptr);
    if (word_start_idx == word_end_idx) {
        AT(p) |= (ptr)(start_mask ^ end_mask);
    } else {

        AT(p) |= ~start_mask;
        p += sizeof(ptr);
        ptr end = p + (word_end_idx - 1) * sizeof(ptr);
        while (p < end) {
            AT(p) = -1;
            p += sizeof(ptr);
        }
        if (end_mask) {
            AT(end) |= end_mask;
        }
    }
}

void
ba_free(bitarray *me) {
    free((void*)me->bits);
    free(me);
}

int
ba_next_unset_bit(bitarray *me, int addr) {
    int word_idx = addr / WORD_BITS;
    int bit_idx = addr & WORD_MASK;

    for (int i = word_idx; i < me->n_words; i++) {
        ptr p = AT(me->bits + i * sizeof(ptr)) >> bit_idx;
        if (~p) {
            int base_bits = i * WORD_BITS;
            int ofs_bits = rightmost_clear_bit(p);
            return base_bits + ofs_bits + bit_idx;
        }
        bit_idx = 0;
    }
    return -1;
}

int
ba_next_set_bit(bitarray *me, int addr) {
    int word_idx = addr / WORD_BITS;
    int bit_idx = addr & WORD_MASK;

    for (int i = word_idx; i < me->n_words; i++) {
        ptr p = AT(me->bits + i * sizeof(ptr)) >> bit_idx;
        if (p) {
            int base_bits = i * WORD_BITS;
            int ofs_bits = rightmost_set_bit(p);
            return base_bits + ofs_bits + bit_idx;
        }
    }
    return -1;
}
