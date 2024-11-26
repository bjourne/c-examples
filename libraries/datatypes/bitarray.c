// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <string.h>
#include "bitarray.h"
#include "bits.h"

#define WORD_MASK   (BA_WORD_BITS - 1)
#define WORD_IDX(i) (i / BA_WORD_BITS)
#define OFS(base, i)  ((base) + (ptr)i * sizeof(ptr))

extern inline int rightmost_clear_bit(ptr x);
extern inline int rightmost_set_bit(ptr x);
extern inline ptr bw_log2(ptr x);

void
ba_clear(bitarray *me) {
    memset((void *)me->bits, 0, (ptr)me->n_words * sizeof(ptr));
}

bitarray *
ba_init(int n_bits) {
    if (n_bits % BA_WORD_BITS != 0) {
        error("ba_init: %d not a multiple of %d", n_bits, BA_WORD_BITS);
    }
    bitarray *me = malloc(sizeof(bitarray));
    me->n_words = n_bits / BA_WORD_BITS;
    me->n_bits = n_bits;
    me->bits = (ptr)malloc((ptr)me->n_words * sizeof(ptr));
    ba_clear(me);
    return me;
}

bool
ba_get_bit(bitarray *me, int addr) {
    int word_idx = addr / BA_WORD_BITS;
    int bit_idx = addr & WORD_MASK;
    ptr p = OFS(me->bits, word_idx);
    return AT(p) & ((ptr)1 << bit_idx);
}

void
ba_set_bit(bitarray *me, int addr) {
    int word_idx = addr / BA_WORD_BITS;
    int bit_idx = addr & WORD_MASK;
    ptr p = OFS(me->bits, word_idx);
    AT(p) |= (ptr)1 << bit_idx;
}

void
ba_set_bit_range(bitarray *me, int addr, int n) {
    int word_start_idx = addr / BA_WORD_BITS;
    int word_end_idx = (addr + n) / BA_WORD_BITS;

    int bit_start_idx = addr & WORD_MASK;
    int bit_end_idx = (addr + n) & WORD_MASK;

    ptr start_mask = ((ptr)1 << bit_start_idx) - (ptr)1;
    ptr end_mask = ((ptr)1 << bit_end_idx) - (ptr)1;

    ptr p = OFS(me->bits, word_start_idx);
    if (word_start_idx == word_end_idx) {
        AT(p) |= (ptr)(start_mask ^ end_mask);
    } else {
        AT(p) |= ~start_mask;
        p += sizeof(ptr);
        ptr end = OFS(me->bits, word_end_idx);
        while (p < end) {
            AT(p) = (ptr)-1;
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
    int word_idx = addr / BA_WORD_BITS;
    int bit_idx = addr & WORD_MASK;

    for (int i = word_idx; i < me->n_words; i++) {
        ptr p = AT(OFS(me->bits, i));
        // Cast it to a signed type so that the shifted in bits are
        // set.
        ptr pattern = (ptr)((intptr_t)p >> bit_idx);
        if (~pattern) {
            int base_bits = i * BA_WORD_BITS;
            int ofs_bits = rightmost_clear_bit(pattern);
            return base_bits + ofs_bits + bit_idx;
        }
        bit_idx = 0;
    }
    return me->n_bits;
}

int
ba_next_set_bit(bitarray *me, int addr) {
    int word_idx = addr / BA_WORD_BITS;
    int bit_idx = addr & WORD_MASK;

    for (int i = word_idx; i < me->n_words; i++) {
        ptr p = AT(OFS(me->bits, i));
        ptr pattern = p >> bit_idx;
        if (pattern) {
            int base_bits = i * BA_WORD_BITS;
            int ofs_bits = rightmost_set_bit(pattern);
            return base_bits + ofs_bits + bit_idx;
        }
        bit_idx = 0;
    }
    return me->n_bits;
}

unsigned int
ba_bitsum(bitarray *me) {
    unsigned int sum = 0;
    for (int i = 0; i < me->n_words; i++) {
        ptr p = AT(OFS(me->bits, i));
        unsigned int count = BIT_COUNT(p);
        sum += count;
    }
    return sum;
}
