#ifndef FILES3D_H
#define FILES3D_H

#include "linalg/linalg.h"

// A simple library for loading 3d files.
#define FILE3D_ERR_NONE               0
#define FILE3D_ERR_FILE_NOT_FOUND     1
#define FILE3D_ERR_UNKNOWN_EXTENSION  2
#define FILE3D_ERR_GEO_FORMAT         3

typedef struct {
    int n_indices;
    int *indices;

    int n_verts;
    vec3 *verts;

    vec3 *normals;
    vec2 *coords;

    // Error
    int error_code;
    char *error_line;
} file3d;

file3d *
f3d_load(char *filename);

void
f3d_free(file3d *me);


#endif
