#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

typedef struct _poly_mesh {
    int n_tris;
    // Three indices per triangle.
    int *indices;
    vec3 *positions;
    vec3 *normals;
    vec2 *coords;
} triangle_mesh;

triangle_mesh *tm_from_file(const char *filename);
void tm_free(triangle_mesh *me);

#endif
