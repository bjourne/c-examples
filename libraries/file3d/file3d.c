// Copyright (C) 2019, 2023 Björn Lindqvist <bjourne@gmail.com>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "file3d.h"
#include "paths/paths.h"

void
f3d_set_error(file3d *me, int error_code, char *error_line) {
    if (error_line) {
        me->error_line = strdup(error_line);
    }
    me->error_code = error_code;
}

file3d *
f3d_load(char *fname) {
    file3d *me = (file3d *)malloc(sizeof(file3d));
    me->verts = NULL;
    me->vertex_indices = NULL;
    me->normals = NULL;
    me->normal_indices = NULL;
    me->coords = NULL;
    me->coord_indices = NULL;
    f3d_set_error(me, FILE3D_ERR_NONE, NULL);
    FILE *f = fopen(fname, "rb");
    if (!f) {
        f3d_set_error(me, FILE3D_ERR_FILE_NOT_FOUND, NULL);
        return me;
    }

    const char *ext = paths_ext(fname);
    if (!strcmp("geo", ext)) {
        f3d_load_geo(me, f);
    } else if (!strcmp("obj", ext) || !strcmp("OBJ", ext)) {
        f3d_load_obj(me, f);
    } else {
        f3d_set_error(me, FILE3D_ERR_UNKNOWN_EXTENSION, NULL);
    }
    fclose(f);
    return me;
}

char *
f3d_get_error_string(file3d *me) {
    switch (me->error_code) {
    case FILE3D_ERR_NONE:
        return "No error";
    case FILE3D_ERR_FILE_NOT_FOUND:
        return "File not found";
    case FILE3D_ERR_OBJ_FACE_VARYING:
        return "Obj: Varying faces detected";
    case FILE3D_ERR_OBJ_LINE_UNPARSABLE:
        return "Obj: Unparsable line";
    default:
        return "Unknown error";
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
    if (me->normal_indices) {
        free(me->normal_indices);
    }
    if (me->normals) {
        free(me->normals);
    }
    if (me->coord_indices) {
        free(me->coord_indices);
    }
    if (me->coords) {
        free(me->coords);
    }
    free(me);
}
