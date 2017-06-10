#ifndef LOADERS_H
#define LOADERS_H

#include <stdbool.h>
#include "linalg/linalg.h"

bool int_read(FILE *f, int *value);
int *int_array_read(FILE *f, int n);
vec2 *v2_array_read(FILE *f, int n);
vec3 *v3_array_read(FILE *f, int n);

bool
load_geo_file(const char *fname,
              int *n_tris, int **indices,
              vec3 **verts, vec3 **normals, vec2 **coords);

#endif
