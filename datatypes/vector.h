#ifndef VECTOR_H
#define VECTOR_H

#include "common.h"

// Growable vector
typedef struct {
    ptr *array;
    size_t size;
    size_t used;
} vector;

vector *v_init(size_t size);
void v_free(vector *v);

void v_add(vector *v, ptr p);
void v_add_all(vector *v, ptr *base, size_t n);
void v_grow(vector *v, size_t req);

ptr v_pop(vector *v);

#endif
