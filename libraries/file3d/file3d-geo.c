#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "file3d/file3d.h"

static bool
int_read(FILE *f, int *value) {
    int ret = fscanf(f, "%d", value);
    return ret == 1;
}

static bool
float_read(FILE *f, float *value) {
    int ret = fscanf(f, "%f", value);
    return ret == 1;
}

static int *
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

static vec2 *
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

static vec3 *
v3_array_read(FILE *f, int n) {
    vec3 *arr = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        if (!v3_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}

// Error reporting is spotty
void
f3d_load_geo(file3d *me, FILE *f) {
    int n_faces;
    if (!int_read(f, &n_faces)) {
        return;
    }
    int *faces = int_array_read(f, n_faces);
    if (!faces) {
        return;
    }
    int n_indices = int_array_sum(faces, n_faces);
    if (n_indices > FILE3D_MAX_N_INDICES) {
        f3d_set_error(me, FILE3D_ERR_TO_BIG, NULL);
        return;
    }

    int *indices = int_array_read(f, n_indices);
    if (!indices)  {
        free(faces);
        return;
    }
    me->n_verts = int_array_max(indices, n_indices) + 1;
    if (me->n_verts > FILE3D_MAX_N_VERTS) {
        free(faces);
        free(indices);
        f3d_set_error(me, FILE3D_ERR_TO_BIG, NULL);
        return;
    }

    me->verts = v3_array_read(f, me->n_verts);
    if (!me->verts) {
        free(faces);
        free(indices);
        return;
    }

    vec3 *normals = v3_array_read(f, n_indices);
    if (!normals) {
        free(faces);
        free(indices);
        return;
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

    me->n_tris = n_tris;
    me->vertex_indices = (int *)malloc(sizeof(int) * n_tris * 3);
    me->n_normals = n_tris * 3;
    me->normals = (vec3 *)malloc(sizeof(vec3) * me->n_normals);
    me->n_coords = n_tris * 3;
    me->coords = (vec2 *)malloc(sizeof(vec2) * me->n_coords);
    for (int i = 0, k = 0, l = 0; i < n_faces; i++) {
        for (int j = 0; j < faces[i] - 2; j++) {
            me->vertex_indices[l] = indices[k];
            me->vertex_indices[l + 1] = indices[k + j + 1];
            me->vertex_indices[l + 2] = indices[k + j + 2];
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

    // Normals aren't shared so we need to fake their indices.
    me->normal_indices = (int *)malloc(sizeof(int) * n_tris * 3);
    for (int i = 0; i < me->n_normals; i++) {
        me->normal_indices[i] = i;
    }

    // Fake coord indexes in the same way.
    me->coord_indices = (int *)malloc(sizeof(int) * n_tris * 3);
    for (int i = 0; i < me->n_coords; i++) {
        me->coord_indices[i] = i;
    }


    free(faces);
    free(indices);
    free(normals);
    free(coords);
}
