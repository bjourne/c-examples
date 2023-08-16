// Copyright (C) 2019, 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <string.h>
#include "common.h"
#include "vector.h"

var_vector *
var_vec_init(size_t capacity, size_t el_size) {
    var_vector *me = malloc(sizeof(var_vector));
    me->array = malloc(el_size * capacity);
    me->capacity = capacity;
    me->used = 0;
    me->el_size = el_size;
    return me;
}

void
var_vec_free(var_vector *me) {
    free(me->array);
    free(me);
}

void
var_vec_grow(var_vector *me, size_t req) {
    size_t s1 = me->capacity + me->capacity / 2;
    size_t new_capacity = MAX(s1, req);
    me->array = realloc(me->array, new_capacity * me->el_size);
    me->capacity = new_capacity;
}

void
var_vec_add(var_vector *me, void *src) {
    me->used++;
    if (me->used > me->capacity) {
        var_vec_grow(me, 0);
    }
    memcpy(me->array + (me->used - 1) * me->el_size, src, me->el_size);
}

void
var_vec_remove(var_vector *me, void *dst) {
    if (me->used == 0) {
        error("Vector underflow!");
    }
    me->used--;
    memcpy(dst, me->array + me->used * me->el_size, me->el_size);
}

void
var_vec_remove_at(var_vector *me, size_t i, void *dst) {
    if (i >= me->used) {
        error("Index out of bounds!");
    }
    void *el_addr = me->array + me->el_size * i;
    memcpy(dst, el_addr, me->el_size);
    me->used--;
    size_t n_copy = (me->used - i) * me->el_size;
    memmove(el_addr, el_addr + me->el_size, n_copy);
}

vector *
v_init(size_t size) {
    vector *v = malloc(sizeof(vector));
    v->array = malloc(NPTRS(size));
    v->size = size;
    v->used = 0;
    return v;
}

void
v_grow(vector *v, size_t req) {
    size_t s1 = v->size + v->size / 2;
    size_t new_size = MAX(s1, req);
    v->array = realloc(v->array, NPTRS(new_size));
    v->size = new_size;
}

void
v_add_all(vector *v, ptr *base, size_t n) {
    size_t req = n + v->used;
    if (req > v->size) {
        v_grow(v, req);
    }
    memcpy(&v->array[v->used], base, NPTRS(n));
    v->used = req;
}

void
v_add(vector *v, ptr p) {
    v->used++;
    if (v->used > v->size) {
        v_grow(v, 0);
    }
    v->array[v->used - 1] = p;
}

ptr v_remove(vector *v) {
    if (v->used == 0) {
        error("Vector underflow!");
    }
    return v->array[--v->used];
}

ptr
v_remove_at(vector *v, size_t i) {
    if (i >= v->used) {
        error("Index out of bounds!");
    }
    ptr el = v->array[i];
    v->used--;
    size_t n_copy = (v->used - i) * sizeof(ptr);
    memmove(&v->array[i], &v->array[i + 1], n_copy);
    return el;
}

ptr v_peek(vector *v) {
    if (v->used == 0) {
        error("Vector underflow!");
    }
    return v->array[v->used - 1];
}

void v_free(vector *v) {
    free(v->array);
    free(v);
}
