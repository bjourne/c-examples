#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "common.h"

typedef struct _triangle_mesh {
    int n_tris;
    // Three indices per triangle.
    int *indices;
    vec3 *verts;
    vec3 *normals;
    vec2 *coords;
#if ISECT_PC_P
    float *precomp;
#endif
} triangle_mesh;

triangle_mesh *tm_from_geo_file(const char *fname);
triangle_mesh *tm_from_obj_file(const char *fname);
void tm_free(triangle_mesh *me);

void tm_get_surface_props(triangle_mesh *me, ray_intersection *ri,
                          vec3 *normal, vec2 *tex_coords);
bool tm_intersect(triangle_mesh *me, vec3 orig, vec3 dir,
                  ray_intersection *ri);

#endif
