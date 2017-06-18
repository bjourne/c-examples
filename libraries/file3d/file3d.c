#include <limits.h>
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

bool
int_read(FILE *f, int *value) {
    int ret = fscanf(f, "%d", value);
    return ret == 1;
}

static bool
float_read(FILE *f, float *value) {
    int ret = fscanf(f, "%f", value);
    return ret == 1;
}

int *
int_array_read(FILE *f, int n) {
    int *arr = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        if (!int_read(f, &arr[i])) {
            return NULL;
        }
    }
    return arr;
}

static int
int_array_sum(int *a, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += a[i];
    }
    return sum;
}

static int
int_array_max(int *a, int len) {
    int max = INT_MIN;
    for (int i = 0; i < len; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    return max;
}

static bool
v2_read(vec2 *vec, FILE *f) {
    if (!float_read(f, &vec->x)) {
        return false;
    }
    if (!float_read(f, &vec->y)) {
        return false;
    }
    return true;
}

vec2 *
v2_array_read(FILE *f, int n) {
    vec2 *arr = (vec2 *)malloc(sizeof(vec2) * n);
    for (int i = 0; i < n; i++) {
        if (!v2_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}

static bool
v3_read(vec3 *vec, FILE *f) {
    if (!float_read(f, &vec->x)) {
        return false;
    }
    if (!float_read(f, &vec->y)) {
        return false;
    }
    if (!float_read(f, &vec->z)) {
        return false;
    }
    return true;
}

vec3 *
v3_array_read(FILE *f, int n) {
    vec3 *arr = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        if (!v3_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}

static void
f3d_set_error(file3d *me, int error_code, char *error_line) {
    me->error_code = error_code;
    me->error_line = error_line;
}

static bool
f3d_load_geo(file3d *me, FILE *f) {
    int n_faces;
    if (!int_read(f, &n_faces)) {
        return false;
    }
    int *faces = int_array_read(f, n_faces);
    if (!faces) {
        return false;
    }
    int n_indices = int_array_sum(faces, n_faces);

    int *indices = int_array_read(f, n_indices);
    if (!indices)  {
        free(faces);
        return false;
    }
    me->n_verts = int_array_max(indices, n_indices) + 1;

    me->verts = v3_array_read(f, me->n_verts);
    if (!me->verts) {
        free(faces);
        free(indices);
        return false;
    }

    vec3 *normals = v3_array_read(f, n_indices);
    if (!normals) {
        free(faces);
        free(indices);
        return false;
    }

    vec2 *coords = v2_array_read(f, n_indices);
    if (!coords) {
        free(faces);
        free(indices);
        free(normals);
    }

    // This part chops the read data into triangles.
    int n_tris = 0;
    for (int i = 0; i < n_faces; i++) {
        n_tris += faces[i] - 2;
    }

    me->n_indices = n_tris * 3;
    me->indices = (int *)malloc(sizeof(int) * me->n_indices);
    me->normals = (vec3 *)malloc(sizeof(vec3) * me->n_indices);
    me->coords = (vec2 *)malloc(sizeof(vec2) * me->n_indices);
    for (int i = 0, k = 0, l = 0; i < n_faces; i++) {
        for (int j = 0; j < faces[i] - 2; j++) {
            me->indices[l] = indices[k];
            me->indices[l + 1] = indices[k + j + 1];
            me->indices[l + 2] = indices[k + j + 2];
            me->normals[l] = normals[k];
            me->normals[l + 1] = normals[k + j + 1];
            me->normals[l + 2] = normals[k + j + 2];
            me->coords[l] = coords[k];
            me->coords[l + 1] = coords[k + j + 1];
            me->coords[l + 2] = coords[k + j + 2];
            l += 3;
        }
        k += faces[i];
    }

    free(faces);
    free(indices);
    free(normals);
    free(coords);
    return true;
}


file3d *
f3d_load(char *filename) {
    file3d *me = (file3d *)malloc(sizeof(file3d));
    me->indices = NULL;
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
    if (me->indices) {
        free(me->indices);
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
