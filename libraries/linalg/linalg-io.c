// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <stdlib.h>

#include "datatypes/bits.h"
#include "datatypes/vector.h"
#include "linalg.h"

typedef union {
    int i;
    float f;
    ptr p;
} int_or_float;

bool
v2_sscanf(char *buf, char *fmt, vector *a) {
    int_or_float x, y;
    if (sscanf(buf, fmt, &x.f, &y.f) != 2) {
        return false;
    }
    v_add(a, x.i);
    v_add(a, y.i);
    return true;
}

bool
v3_sscanf(char *buf, const char *fmt, vector *a) {
    int_or_float x, y, z;
    if (sscanf(buf, fmt, &x.f, &y.f, &z.f) != 3) {
        return false;
    }
    v_add(a, x.i);
    v_add(a, y.i);
    v_add(a, z.i);
    return true;
}

vec2 *
v2_array_pack(vector *a) {
    size_t n = a->used / 2;
    vec2 *out = (vec2 *)malloc(sizeof(vec2) * n);
    for (int i  = 0; i < n; i++) {
        out[i].x = BW_PTR_TO_FLOAT(a->array[2 * i]);
        out[i].y = BW_PTR_TO_FLOAT(a->array[2 * i + 1]);
    }
    return out;
}

vec3 *
v3_array_pack(vector *a) {
    size_t n = a->used / 3;
    vec3 *out = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        out[i].x = BW_PTR_TO_FLOAT(a->array[3 * i]);
        out[i].y = BW_PTR_TO_FLOAT(a->array[3 * i + 1]);
        out[i].z = BW_PTR_TO_FLOAT(a->array[3 * i + 2]);
    }
    return out;
}

vec2*
v2_array_read(FILE *f, int *n) {
    char buf[1024];

    if (!fgets(buf, 1024, f)) {
        return NULL;
    }
    *n = atoi(buf);
    vector *coords = v_init(*n);
    vec2 *arr = NULL;
    for (int i = 0; i < *n; i++) {
        if (!fgets(buf, 1024, f)) {
            goto end;
        }
        if (!v2_sscanf(buf, "%f %f", coords)) {
            goto end;
        }
    }
    arr = v2_array_pack(coords);
 end:
    v_free(coords);
    return arr;
}
