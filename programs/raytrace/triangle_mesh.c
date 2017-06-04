#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"

#include "intersection.h"
#include "triangle_mesh.h"

static bool
read_float(FILE *f, float *value) {
    int ret = fscanf(f, "%f", value);
    return ret == 1;
}

static bool
read_int(FILE *f, int *value) {
    int ret = fscanf(f, "%d", value);
    return ret == 1;
}

static bool
vec2_read(vec2 *vec, FILE *f) {
    if (!read_float(f, &vec->x)) {
        return false;
    }
    if (!read_float(f, &vec->y)) {
        return false;
    }
    return true;
}

static vec2 *
vec2_array_read(FILE *f, int n) {
    vec2 *arr = (vec2 *)malloc(sizeof(vec2) * n);
    for (int i = 0; i < n; i++) {
        if (!vec2_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}

static bool
read_int_array(FILE *f, int n, int *ptr) {
    for (int i = 0; i < n; i++) {
        int val;
        if (!read_int(f, &val)) {
            return false;
        }
        ptr[i] = val;
    }
    return true;
}

static bool
vec3_read(vec3 *vec, FILE *f) {
    if (!read_float(f, &vec->x)) {
        return false;
    }
    if (!read_float(f, &vec->y)) {
        return false;
    }
    if (!read_float(f, &vec->z)) {
        return false;
    }
    return true;
}

static vec3 *
vec3_array_read(FILE *f, int n) {
    vec3 *arr = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        if (!vec3_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}
#if defined(ISECT_PRECOMP12)
static void
tm_precomp12(triangle_mesh *me, int idx, vec3 v0, vec3 v1, vec3 v2) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);

    // Build transform from global to barycentric coordinates.
    float x1, x2;
    float num = v3_dot(v0, n);
    float *T = &me->precomp12[idx*12];
    if (fabs(n.x) > fabs(n.y) && fabs(n.x) > fabs(n.z)) {
        // x is pivot
        x1 = v1.y * v0.z - v1.z * v0.y;
        x2 = v2.y * v0.z - v2.z * v0.y;

        T[0] = 0.0f;
        T[1] = e2.z / n.x;
        T[2] = -e2.y / n.x;
        T[3] = x2 / n.x;

        T[4] = 0.0f;
        T[5] = -e1.z / n.x;
        T[6] = e1.y / n.x;
        T[7] = -x1 / n.x;

        T[8] = 1.0f;
        T[9] = n.y / n.x;
        T[10] = n.z / n.x;
        T[11] = -num / n.x;
    } else if (fabs(n.y) > fabs(n.z)) {
        // y is pivot
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;

        T[0] = -e2.z / n.y;
        T[1] = 0.0f;
        T[2] = e2.x / n.y;
        T[3] = x2 / n.y;

        T[4] = e1.z / n.y;
        T[5] = 0.0f;
        T[6] = -e1.x / n.y;
        T[7] = -x1 / n.y;

        T[8] = n.x / n.y;
        T[9] = 1.0f;
        T[10] = n.z / n.y;
        T[11] = -num / n.y;
    } else if (fabs(n.z) > 0.0f) {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;

        T[0] = e2.y / n.z;
        T[1] = -e2.x / n.z;
        T[2] = 0.0f;
        T[3] = x2 / n.z;

        T[4] = -e1.y / n.z;
        T[5] = e1.x / n.z;
        T[6] = 0.0f;
        T[7] = -x1 / n.z;

        T[8] = n.x / n.z;
        T[9] = n.y / n.z;
        T[10] = 1.0f;
        T[11] = -num / n.z;
    } else {
        error("Impossible!");
    }
}
#endif

triangle_mesh *
tm_init(int n_faces,
        int *faces,
        int *verts_indices,
        vec3 *verts,
        vec3 *normals,
        vec2 *coords) {
    triangle_mesh *me = (triangle_mesh *)malloc(sizeof(triangle_mesh));
    me->n_tris = 0;
    int n_verts = 0;
    int k = 0;
    for (int i = 0; i < n_faces; i++) {
        me->n_tris += faces[i] - 2;
        for (int j = 0; j < faces[i]; j++) {
            if (verts_indices[k + j] > n_verts) {
                n_verts = verts_indices[k + j];
            }
        }
        k += faces[i];
    }
    n_verts++;

    me->positions = (vec3 *)malloc(sizeof(vec3) * n_verts);
    for (int i = 0; i < n_verts; i++) {
        me->positions[i] = verts[i];
    }

    // Allocate memory to store triangle indices.
    me->indices = (int *)malloc(sizeof(int) * me->n_tris * 3);
    me->normals = (vec3 *)malloc(sizeof(vec3) * me->n_tris * 3);
    me->coords = (vec2 *)malloc(sizeof(vec2) * me->n_tris * 3);

    for (int i = 0, k = 0, l = 0; i < n_faces; i++) {
        for (int j = 0; j < faces[i] - 2; j++) {
            me->indices[l] = verts_indices[k];
            me->indices[l + 1] = verts_indices[k + j + 1];
            me->indices[l + 2] = verts_indices[k + j + 2];
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

#if defined(ISECT_PRECOMP12)
    me->precomp12 = (float *)malloc(sizeof(float) * 12 * me->n_tris);
    int *at_idx = me->indices;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = me->positions[*at_idx++];
        vec3 v1 = me->positions[*at_idx++];
        vec3 v2 = me->positions[*at_idx++];
        tm_precomp12(me, i, v0, v1, v2);
    }
#endif
    return me;
}

void
tm_free(triangle_mesh *me) {
    free(me->indices);
    free(me->normals);
    free(me->coords);
    free(me->positions);
#if defined(ISECT_PRECOMP12)
    free(me->precomp12);
#endif
    free(me);
}

triangle_mesh *
tm_from_file(const char *filename) {
    FILE* f = fopen(filename, "rb");
    int *faces = NULL;
    int *verts_indices = NULL;
    vec3 *verts = NULL;
    vec3 *normals = NULL;
    vec2 *coords = NULL;
    triangle_mesh *tm = NULL;
    if (!f) {
        goto end;
    }

    // Read polygon count
    int n_faces;
    if (!read_int(f, &n_faces)) {
        goto end;
    }

    faces = (int *)malloc(sizeof(int) * n_faces);
    if (!read_int_array(f, n_faces, faces)) {
        goto end;
    }
    int n_verts_indices = 0;
    for (int i = 0; i < n_faces; i++) {
        n_verts_indices += faces[i];
    }
    verts_indices = (int *)malloc(sizeof(int) * n_verts_indices);
    if (!read_int_array(f, n_verts_indices, verts_indices)) {
        goto end;
    }
    int n_verts_array = 0;
    for (int i = 0; i < n_verts_indices; i++) {
        if (verts_indices[i] > n_verts_array) {
            n_verts_array = verts_indices[i];
        }
    }
    n_verts_array++;

    // Reading vertices
    verts = vec3_array_read(f, n_verts_array);
    if (!verts) {
        goto end;
    }

    // Reading normals
    normals = vec3_array_read(f, n_verts_indices);
    if (!normals) {
        goto end;
    }

    // Reading texture coords
    coords = vec2_array_read(f, n_verts_indices);
    if (!coords) {
        goto end;
    }

    tm = tm_init(n_faces,
                 faces,
                 verts_indices,
                 verts,
                 normals,
                 coords);
 end:
    if (faces) {
        free(faces);
    }
    if (verts_indices) {
        free(verts_indices);
    }
    if (verts) {
        free(verts);
    }
    if (normals) {
        free(normals);
    }
    if (coords) {
        free(coords);
    }
    if (f) {
        fclose(f);
    }
    return tm;
}

void
tm_print_index(triangle_mesh *me, int index) {
    vec3 vec = me->positions[index];
    printf("%d = {%.2f, %.2f, %.2f}", index, vec.x, vec.y, vec.z);
}

void
tm_print(triangle_mesh *me) {
    printf("# triangles: %d\n", me->n_tris);
    int *at_idx = me->indices;
    for (int i = 0; i < me->n_tris; i++) {
        printf("{");
        tm_print_index(me, *at_idx++);
        printf(", ");
        tm_print_index(me, *at_idx++);
        printf(", ");
        tm_print_index(me, *at_idx++);
        printf("}\n");
    }
}

void
tm_get_surface_props(triangle_mesh *me, ray_intersection *ri,
                     vec3 *normal, vec2 *tex_coords) {
    int t0 = 3 * ri->tri_idx;
    int t1 = 3 * ri->tri_idx + 1;
    int t2 = 3 * ri->tri_idx + 2;

    // Texture coordinates
    vec2 st0 = me->coords[t0];
    vec2 st1 = me->coords[t1];
    vec2 st2 = me->coords[t2];
    vec2 st0_scaled = v2_scale(st0, 1 - ri->uv.x - ri->uv.y);
    vec2 st1_scaled = v2_scale(st1, ri->uv.x);
    vec2 st2_scaled = v2_scale(st2, ri->uv.y);
    *tex_coords = v2_add(v2_add(st0_scaled, st1_scaled), st2_scaled);

    vec3 n0 = me->normals[t0];
    vec3 n1 = me->normals[t1];
    vec3 n2 = me->normals[t2];
    vec3 n0_scaled = v3_scale(n0, 1 - ri->uv.x - ri->uv.y);
    vec3 n1_scaled = v3_scale(n1, ri->uv.x);
    vec3 n2_scaled = v3_scale(n2, ri->uv.y);
    *normal = v3_add(v3_add(n0_scaled, n1_scaled), n2_scaled);
}

bool
tm_intersect(triangle_mesh *me, vec3 orig, vec3 dir,
             ray_intersection *ri) {
    float nearest = FLT_MAX;
    int *at_idx = me->indices;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = me->positions[*at_idx++];
        vec3 v1 = me->positions[*at_idx++];
        vec3 v2 = me->positions[*at_idx++];
        float u, v, t;

#if defined(ISECT_MT)
        bool isect = moeller_trumbore_isect(orig, dir,
                                            v0, v1, v2,
                                            &t, &u, &v);
#elif defined(ISECT_PRECOMP12)
        float *trans = &me->precomp12[i*12];
        bool isect = precomp12_isect(orig, dir,
                                     v0, v1, v2,
                                     &t, &u, &v, trans);
#endif
        if (isect && t < nearest) {
            nearest = t;
            ri->t = t;
            ri->uv.x = u;
            ri->uv.y = v;
            ri->tri_idx = i;
        }
    }
    return nearest < FLT_MAX;
}
