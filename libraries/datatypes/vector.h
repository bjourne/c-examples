// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
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

ptr v_remove(vector *v);
ptr v_remove_at(vector *v, size_t i);
ptr v_peek(vector *v);

// This class works like vector but the user can specify the type of
// the elements stored.
typedef struct {
    void *array;
    size_t capacity;
    size_t used;
    size_t el_size;
} var_vector;


var_vector *var_vec_init(size_t capacity, size_t el_size);
void var_vec_free(var_vector *me);
void var_vec_add(var_vector *me, void *el);
void var_vec_remove(var_vector *me, void *el);

#endif
