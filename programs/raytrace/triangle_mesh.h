#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

typedef struct _triangle_mesh {
    int n_tris;
    // Three indices per triangle.
    int *indices;
    vec3 *positions;
    vec3 *normals;
    vec2 *coords;
#if defined(ISECT_PRECOMP12)
    float *precomp12;
#endif
} triangle_mesh;

triangle_mesh *tm_from_file(const char *filename);
void tm_free(triangle_mesh *me);

#endif
