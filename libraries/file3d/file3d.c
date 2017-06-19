#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "file3d.h"

// Utility
static char *
fname_ext(char *fname) {
    char *dot = strrchr(fname, '.');
    if(!dot || dot == fname)
        return "";
    return dot + 1;
}


void
f3d_set_error(file3d *me, int error_code, char *error_line) {
    if (error_line) {
        me->error_line = strdup(error_line);
    }
    me->error_code = error_code;
}

file3d *
f3d_load(char *filename) {
    file3d *me = (file3d *)malloc(sizeof(file3d));
    me->vertex_indices = NULL;
    me->verts = NULL;
    me->normals = NULL;
    me->coords = NULL;
    f3d_set_error(me, FILE3D_ERR_NONE, NULL);
    FILE *f = fopen(filename, "rb");
    if (!f) {
        f3d_set_error(me, FILE3D_ERR_FILE_NOT_FOUND, NULL);
        return me;
    }

    char *ext = fname_ext(filename);
    if (!strcmp("geo", ext)) {
        if (!f3d_load_geo(me, f)) {
            f3d_set_error(me, FILE3D_ERR_GEO_FORMAT, NULL);
        }
    } else if (!strcmp("obj", ext) || !strcmp("OBJ", ext)) {
        if (!f3d_load_obj(me, f)) {
            f3d_set_error(me, FILE3D_ERR_OBJ_FORMAT, NULL);
        }

    } else {
        f3d_set_error(me, FILE3D_ERR_UNKNOWN_EXTENSION, NULL);
    }
    fclose(f);
    return me;
}

char *
f3d_get_error_string(file3d *me) {
    if (me->error_code == FILE3D_ERR_FILE_NOT_FOUND) {
        return "File not found";
    } else {
        return "No error";
    }
}

void
f3d_free(file3d *me) {
    if (me->vertex_indices) {
        free(me->vertex_indices);
    }
    if (me->verts) {
        free(me->verts);
    }
    if (me->normals) {
        free(me->normals);
    }
    if (me->coords) {
        free(me->coords);
    }
    free(me);
}
