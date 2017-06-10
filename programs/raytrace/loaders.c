#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loaders.h"

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

bool
load_geo_file(const char *fname,
              int *n_tris, int **indices,
              vec3 **verts, vec3 **normals, vec2 **coords) {
    FILE* f = fopen(fname, "rb");
    int *t_faces = NULL;
    int *t_indices = NULL;
    vec3 *t_verts = NULL;
    vec3 *t_normals = NULL;
    vec2 *t_coords = NULL;
    bool ret = false;
    if (!f) {
        goto end;
    }

    // Read polygon count
    int n_faces;
    if (!int_read(f, &n_faces)) {
        goto end;
    }

    t_faces = int_array_read(f, n_faces);
    if (!t_faces) {
        goto end;
    }
    int n_indices = 0;
    for (int i = 0; i < n_faces; i++) {
        n_indices += t_faces[i];
    }
    t_indices = int_array_read(f, n_indices);
    if (!t_indices) {
        goto end;
    }
    int n_verts = 0;
    for (int i = 0; i < n_indices; i++) {
        if (t_indices[i] > n_verts) {
            n_verts = t_indices[i];
        }
    }
    n_verts++;

    // Reading vertices
    t_verts = v3_array_read(f, n_verts);
    if (!t_verts) {
        goto end;
    }

    // Reading normals
    t_normals = v3_array_read(f, n_indices);
    if (!t_normals) {
        goto end;
    }

    // Reading texture coords
    t_coords = v2_array_read(f, n_indices);
    if (!t_coords) {
        goto end;
    }

    // This part chops the read data into triangles.
    *n_tris = 0;
    for (int i = 0; i < n_faces; i++) {
        *n_tris += t_faces[i] - 2;
    }

    *verts = (vec3 *)malloc(sizeof(vec3) * n_verts);
    memcpy(*verts, t_verts, sizeof(vec3) * n_verts);

    *indices = (int *)malloc(sizeof(int) * *n_tris * 3);
    *normals = (vec3 *)malloc(sizeof(vec3) * *n_tris * 3);
    *coords = (vec2 *)malloc(sizeof(vec2) * *n_tris * 3);

    for (int i = 0, k = 0, l = 0; i < n_faces; i++) {
        for (int j = 0; j < t_faces[i] - 2; j++) {
            (*indices)[l] = t_indices[k];
            (*indices)[l + 1] = t_indices[k + j + 1];
            (*indices)[l + 2] = t_indices[k + j + 2];
            (*normals)[l] = t_normals[k];
            (*normals)[l + 1] = t_normals[k + j + 1];
            (*normals)[l + 2] = t_normals[k + j + 2];
            (*coords)[l] = t_coords[k];
            (*coords)[l + 1] = t_coords[k + j + 1];
            (*coords)[l + 2] = t_coords[k + j + 2];
            l += 3;
        }
        k += t_faces[i];
    }
    ret = true;
 end:
    if (t_faces) {
        free(t_faces);
    }
    if (t_indices) {
        free(t_indices);
    }
    if (t_verts) {
        free(t_verts);
    }
    if (t_normals) {
        free(t_normals);
    }
    if (t_coords) {
        free(t_coords);
    }
    if (f) {
        fclose(f);
    }
    return ret;
}
