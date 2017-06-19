#ifndef FILES3D_H
#define FILES3D_H

#include <stdio.h>
#include "linalg/linalg.h"

// A simple library for loading 3d files.
#define FILE3D_ERR_NONE                 0
#define FILE3D_ERR_FILE_NOT_FOUND       1
#define FILE3D_ERR_UNKNOWN_EXTENSION    2
#define FILE3D_ERR_GEO_FORMAT           3
#define FILE3D_ERR_OBJ_FORMAT           4
#define FILE3D_ERR_OBJ_LINE_UNPARSABLE  5

typedef struct {
    int n_tris;
    int *vertex_indices;
    int *normal_indices;
    int *coord_indices;

    int n_verts;
    vec3 *verts;

    int n_normals;
    vec3 *normals;

    int n_coords;
    vec2 *coords;

    // Error
    int error_code;
    char *error_line;
} file3d;

// Private
bool f3d_load_geo(file3d *me, FILE *f);
bool f3d_load_obj(file3d *me, FILE *f);
void f3d_set_error(file3d *me, int error_code, char *error_line);

// Public
file3d *f3d_load(char *filename);
void f3d_free(file3d *me);
char *f3d_get_error_string(file3d *me);


#endif
