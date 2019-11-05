// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#ifndef LINALG_IO_H
#define LINALG_IO_H

#include <stdio.h>

#include "datatypes/vector.h"
#include "linalg/linalg.h"

// Uses the format string to scan for a 2d or 3d vector to add to the
// (resizable) array.
bool v2_sscanf(char *buf, const char *fmt, vector *a);
bool v3_sscanf(char *buf, const char *fmt, vector *a);

// Takes a vector containing 2n floats and returns an array of vec2.
vec2 *v2_array_pack(vector *a);
vec3 *v3_array_pack(vector *a);

vec2 *v2_array_read(FILE *f, int *n);

#endif
