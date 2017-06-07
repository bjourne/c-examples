#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "datatypes/vector.h"

#include "isect/isect.h"
#include "linalg/linalg.h"
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

static int *
int_array_read(FILE *f, int n) {
    int *arr = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        if (!read_int(f, &arr[i])) {
            return NULL;
        }
    }
    return arr;
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

void
tm_intersect_precompute(triangle_mesh *me) {
#if ISECT_PC_P
    int bytes = ISECT_PC_N_ELS * sizeof(float) * me->n_tris;
    me->precomp = (float *)malloc(bytes);
    int *at_idx = me->indices;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = me->positions[*at_idx++];
        vec3 v1 = me->positions[*at_idx++];
        vec3 v2 = me->positions[*at_idx++];
        float *addr = &me->precomp[i * ISECT_PC_N_ELS];
        #if ISECT_METHOD == ISECT_PC12 || ISECT_METHOD == ISECT_PC12_B
        isect_precomp12_pre(v0, v1, v2, addr);
        #elif ISECT_METHOD == ISECT_PC9 || ISECT_METHOD == ISECT_PC9_B
        isect_precomp9_pre(v0, v1, v2, addr);
        #endif
    }
#endif
}

triangle_mesh *
tm_init_simple(int n_tris, int *indices, int n_verts, vec3 *verts) {
    triangle_mesh *me = (triangle_mesh *)malloc(sizeof(triangle_mesh));
    me->n_tris = n_tris;
    int ibuf_size = sizeof(int) * me->n_tris * 3;
    me->indices = (int *)malloc(ibuf_size);
    memcpy(me->indices, indices, ibuf_size);

    int vbuf_size = sizeof(vec3) * n_verts;
    me->positions = (vec3 *)malloc(vbuf_size);
    memcpy(me->positions, verts, vbuf_size);

    me->normals = NULL;
    me->coords = NULL;

    tm_intersect_precompute(me);

    return me;
}

triangle_mesh *
tm_init_from_face_list(int n_faces,
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
    tm_intersect_precompute(me);
    return me;
}

void
tm_free(triangle_mesh *me) {
    free(me->indices);
    free(me->normals);
    free(me->coords);
    free(me->positions);
#if ISECT_PC_P
    free(me->precomp);
#endif
    free(me);
}

typedef union {
    ptr i;
    float f;
} u;

// Don't run on untrusted data :)
triangle_mesh *
tm_from_obj_file(const char *fname) {
    FILE* f = fopen(fname, "rb");
    vector *tmp_verts = NULL;
    vector *tmp_indices = NULL;
    triangle_mesh *tm = NULL;
    if (!f) {
        goto end;
    }
    char buf[1024];
    tmp_verts = v_init(10);
    tmp_indices = v_init(10);
    while (fgets(buf, 1024, f)) {
        if (!strncmp(buf, "v ", 2)) {
            u x, y, z;
            if (sscanf(buf, "v %f %f %f", &x.f, &y.f, &z.f) != 3) {
                goto end;
            }
            v_add(tmp_verts, x.i);
            v_add(tmp_verts, y.i);
            v_add(tmp_verts, z.i);
        } else if (!strncmp(buf, "#", 1)) {
            // Skip comments
        } else if (!strncmp(buf, "f ", 2)) {
            // Only supports one face format for now
            int i0, i1, i2;
            if (sscanf(buf, "f %d %d %d", &i0, &i1, &i2) != 3) {
                goto end;
            }
            v_add(tmp_indices, i0 - 1);
            v_add(tmp_indices, i1 - 1);
            v_add(tmp_indices, i2 - 1);
        } else {
            error("Unparsable line '%s'!\n", buf);
        }
    }
    size_t n_tris = tmp_indices->used / 3;
    size_t n_verts = tmp_verts->used / 3;

    // Convert the arrays to right format.
    vec3 *verts = (vec3 *)malloc(sizeof(vec3) * n_verts);
    for (int i = 0; i < n_verts; i++) {
        verts[i].x = ((u)tmp_verts->array[i*3]).f;
        verts[i].y = ((u)tmp_verts->array[i*3+1]).f;
        verts[i].z = ((u)tmp_verts->array[i*3+2]).f;
        verts[i] = v3_scale(verts[i], 70.0f);
    }
    int *indices = (int *)malloc(sizeof(int) * n_tris * 3);
    for (int i = 0; i < n_tris * 3; i++) {
        indices[i] = tmp_indices->array[i];
    }
    tm = tm_init_simple(n_tris, indices, n_verts, verts);
    free(indices);
    free(verts);

 end:
    if (f) {
        fclose(f);
    }
    if (tmp_verts) {
        v_free(tmp_verts);
    }
    if (tmp_indices) {
        v_free(tmp_indices);
    }
    return tm;
}

triangle_mesh *
tm_from_geo_file(const char *fname) {
    FILE* f = fopen(fname, "rb");
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

    faces = int_array_read(f, n_faces);
    if (!faces) {
        goto end;
    }
    int n_verts_indices = 0;
    for (int i = 0; i < n_faces; i++) {
        n_verts_indices += faces[i];
    }
    verts_indices = int_array_read(f, n_verts_indices);
    if (!verts_indices) {
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

    tm = tm_init_from_face_list(n_faces,
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
    if (me->coords) {
        vec2 st0 = me->coords[t0];
        vec2 st1 = me->coords[t1];
        vec2 st2 = me->coords[t2];
        vec2 st0_scaled = v2_scale(st0, 1 - ri->uv.x - ri->uv.y);
        vec2 st1_scaled = v2_scale(st1, ri->uv.x);
        vec2 st2_scaled = v2_scale(st2, ri->uv.y);
        *tex_coords = v2_add(v2_add(st0_scaled, st1_scaled), st2_scaled);
    } else {
        *tex_coords = ri->uv;
    }
    if (me->normals) {
        vec3 n0 = me->normals[t0];
        vec3 n1 = me->normals[t1];
        vec3 n2 = me->normals[t2];
        vec3 n0_scaled = v3_scale(n0, 1 - ri->uv.x - ri->uv.y);
        vec3 n1_scaled = v3_scale(n1, ri->uv.x);
        vec3 n2_scaled = v3_scale(n2, ri->uv.y);
        *normal = v3_add(v3_add(n0_scaled, n1_scaled), n2_scaled);
    } else {
        vec3 v0 = me->positions[me->indices[t0]];
        vec3 v1 = me->positions[me->indices[t1]];
        vec3 v2 = me->positions[me->indices[t2]];
        vec3 e1 = v3_sub(v1, v0);
        vec3 e2 = v3_sub(v2, v0);
        *normal = v3_normalize(v3_cross(e1, e2));
    }
}

bool
tm_intersect(triangle_mesh *me, vec3 o, vec3 d, ray_intersection *ri) {
    float nearest = FLT_MAX;
    float t = 0;
    vec2 uv;
#if ISECT_PC_P
    for (int i = 0; i < me->n_tris; i++) {
        float *T = &me->precomp[i*ISECT_PC_N_ELS];
        bool isect = ISECT_FUN(o, d, &t, &uv, T);
        if (isect && t < nearest) {
            nearest = t;
            ri->t = t;
            ri->uv = uv;
            ri->tri_idx = i;
        }
    }
#else
    int *at_idx = me->indices;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = me->positions[*at_idx++];
        vec3 v1 = me->positions[*at_idx++];
        vec3 v2 = me->positions[*at_idx++];
        bool isect = ISECT_FUN(o, d, v0, v1, v2, &t, &uv);
        if (isect && t < nearest) {
            nearest = t;
            ri->t = t;
            ri->uv = uv;
            ri->tri_idx = i;
        }
    }
#endif
    return nearest < FLT_MAX;
}
