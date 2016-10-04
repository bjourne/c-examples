#include <stdio.h>
#include "heap.h"

// Borrowed from: https://gist.github.com/martinkunev/1365481

#define CMP(a, b) ((a) >= (b))

void hp_add(vector *v, ptr p) {
    if (v->used >= v->size) {
        v_grow(v, 0);
    }
    size_t parent, i;
    for (i = v->used++; i; i = parent) {
        parent = (i - 1) / 2;
        if (CMP(v->array[parent], p))
            break;
        v->array[i] = v->array[parent];
    }
    v->array[i] = p;
}

ptr hp_pop(vector *v) {
    ptr temp = v_pop(v);
    ptr max = v->array[0];
    size_t i, swap, other;
    for (i = 0; 1; i = swap) {
        swap = (i * 2) + 1;
        if (swap >= v->used)
            break;
        other = swap + 1;
        if ((other < v->used) && CMP(v->array[other], v->array[swap]))
            swap = other;
        if (CMP(temp, v->array[swap]))
            break;
        v->array[i] = v->array[swap];
    }
    v->array[i] = temp;
    return max;
}
