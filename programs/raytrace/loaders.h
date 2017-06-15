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
              int *n_verts, vec3 **verts,
              vec3 **normals, vec2 **coords);
bool
load_obj_file(const char *fname,
              int *n_tris, int **v_indices, int **n_indices,
              int *n_verts, vec3 **verts,
              int *n_normals, vec3 **normals,
              char *err_buf);

// Parameters:
//   - [in]fname: filename to read from
//   - [out]n_tris: number of triangles
//   - [out]v_indices: array of vertex indices
//   - [out]n_indices: array of normal indices (or NULL if there weren't any)
//   - [out]n_verts: number of vertices
//   - [out]verts: array of vertex data
//   - [out]n_normals: number of normals
//   - [out]normals: array of normal data
//   - [out]coords: array of coordinate data
//   - [out]err_buf: error message, if loading failed
//
// Returns: true if loading succeeded, false otherwise.
bool
load_any_file(const char *fname,
              int *n_tris, int **v_indices, int **n_indices,
              int *n_verts, vec3 **verts,
              int *n_normals, vec3 **normals,
              vec2 **coords,
              char *err_buf);

#endif
