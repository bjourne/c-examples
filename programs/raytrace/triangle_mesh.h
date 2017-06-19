#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "common.h"

typedef struct _triangle_mesh {
    int n_tris;
    vec3 *verts;
    vec3 *normals;
    vec2 *coords;
#if ISECT_PC_P
    ISECT_DATA *precomp;
#endif
} triangle_mesh;

triangle_mesh *tm_from_file(char *fname, float scale, vec3 translate);
void tm_free(triangle_mesh *me);

void tm_get_surface_props(triangle_mesh *me, ray_intersection *ri,
                          vec3 *normal, vec2 *tex_coords);
bool tm_intersect(triangle_mesh *me, vec3 orig, vec3 dir,
                  ray_intersection *ri);

#endif
