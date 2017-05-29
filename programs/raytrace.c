// This is a raytracer written in C.
// This code is verrrry much based on: www.scratchapixel.com
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "datatypes/common.h"

float
deg_to_rad(const float deg) {
    return deg * M_PI / 180;
}

bool
read_int(FILE *f, int *value) {
    int ret = fscanf(f, "%d", value);
    return ret == 1;
}

bool
read_float(FILE *f, float *value) {
    int ret = fscanf(f, "%f", value);
    return ret == 1;
}

typedef struct _vec3 {
    float x, y, z;
} vec3;

vec3
vec3_sub(vec3 l, vec3 r) {
    return (vec3){l.x - r.x, l.y - r.y, l.z - r.z};
}

vec3
vec3_cross(vec3 l, vec3 r) {
    vec3 ret = {
        l.y * r.z - l.z * r.y,
        l.z * r.x - l.x * r.z,
        l.x * r.y - l.y * r.x
    };
    return ret;
}

float
vec3_dot(vec3 l, vec3 r) {
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

bool
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

vec3
vec3_normalize(vec3 in) {
    vec3 out = in;
    float norm = in.x * in.x + in.y * in.y + in.z * in.z;
    if (norm > 0) {
        float factor = 1.0f / sqrt(norm);
        out.x *= factor;
        out.y *= factor;
        out.z *= factor;
    }
    return out;
}

vec3 *
vec3_array_read(FILE *f, int n) {
    vec3 *arr = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        if (!vec3_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}

static const float k_epsilon = 1e-8;

bool
ray_tri_intersect(vec3 orig, vec3 dir,
                  vec3 v0, vec3 v1, vec3 v2,
                  float *t, float *u, float *v) {
    vec3 v0v1 = vec3_sub(v1, v0);
    vec3 v0v2 = vec3_sub(v2, v1);
    vec3 pvec = vec3_cross(dir, v0v2);
    float det = vec3_dot(v0v1, pvec);

    if (fabs(det) < k_epsilon) {
        return false;
    }
    float inv_det = 1 / det;
    vec3 tvec = vec3_sub(orig, v0);
    *u = vec3_dot(tvec, pvec) * inv_det;
    if (*u < 0 || *u > 1) {
        return false;
    }
    vec3 qvec = vec3_cross(tvec, v0v1);
    *v = vec3_dot(dir, qvec) * inv_det;
    if (*v < 0 || *u + *v > 1) {
        return false;
    }
    *t = vec3_dot(v0v2, qvec) * inv_det;
    return true;
}

bool
vec3_array_to_ppm(vec3 *arr, const char *filename,
                  int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        return false;
    }
    if (fprintf(f, "P6\n%d %d\n255\n", width, height) <= 0) {
        return false;
    }
    for (int i = 0; i < width * height; i++) {
        int r = 255 * CLAMP(0, 1, arr[i].x);
        int g = 255 * CLAMP(0, 1, arr[i].y);
        int b = 255 * CLAMP(0, 1, arr[i].z);
        if (fprintf(f, "%c%c%c", r, g, b) <= 0) {
            return false;
        }
    }
    fclose(f);
    return true;
}

typedef struct _vec2 {
    float x;
    float y;
} vec2;

bool
vec2_read(vec2 *vec, FILE *f) {
    if (!read_float(f, &vec->x)) {
        return false;
    }
    if (!read_float(f, &vec->y)) {
        return false;
    }
    return true;
}

vec2 *
vec2_array_read(FILE *f, int n) {
    vec2 *arr = (vec2 *)malloc(sizeof(vec2) * n);
    for (int i = 0; i < n; i++) {
        if (!vec2_read(&arr[i], f)) {
            return NULL;
        }
    }
    return arr;
}




bool
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

typedef struct _poly_mesh {
    int n_tris;
    // Three indices per triangle.
    int *indices;
    vec3 *positions;
    vec3 *normals;
    vec2 *coords;
} triangle_mesh;

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

    return me;
}

void
tm_free(triangle_mesh *me) {
    free(me->indices);
    free(me->normals);
    free(me->coords);
    free(me->positions);
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
    fclose(f);
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

bool
tm_intersect(triangle_mesh *me, vec3 orig, vec3 dir) {
    int *at_idx = me->indices;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = me->positions[*at_idx++];
        vec3 v1 = me->positions[*at_idx++];
        vec3 v2 = me->positions[*at_idx++];
        float t, u, v;
        if (ray_tri_intersect(orig, dir,
                              v0, v1, v2,
                              &t, &u, &v)) {
            printf("%.2f\n", t);
            return true;
        }
    }
    return false;
}

bool
trace(vec3 orig, vec3 dir, triangle_mesh *tm) {
    if (tm_intersect(tm, orig, dir)) {
        return true;
    }
    return false;
}

vec3
cast_ray(vec3 orig, vec3 dir, triangle_mesh *tm) {
    vec3 col = {0};
    if (trace(orig, dir, tm)) {
        col.x = 1.0;
    }
    return col;
}

void
render(triangle_mesh *tm, vec3 *fbuf, float fov, int width, int height) {
    float scale = tan(deg_to_rad(fov * 0.5));
    float image_aspect_ratio = (float)width / (float)height;
    vec3 orig = {24.49, 24.01, 22.17};
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float ray_x = (2 * (x + 0.5) / (float)width - 1)
                * image_aspect_ratio * scale;
            float ray_y = (1 - 2 * (y + 0.5) / (float)height)
                * scale;
            vec3 dir = {ray_x, ray_y, -1};
            dir = vec3_normalize(dir);
            *fbuf = cast_ray(orig, dir, tm);
            fbuf++;
        }
    }
}

void
usage() {
    printf("usage: raytrace mesh-file width height image\n");
    exit(1);
}

int
main(int argc, char *argv[]) {
    if (argc != 5) {
        usage();
    }
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    if (width <= 0 || width > 2048 ||
        height <= 0 || height > 2048) {
        usage();
    }
    char *inf = argv[1];
    char *outf = argv[4];
    triangle_mesh *tm = tm_from_file(inf);
    if (!tm) {
        error("Failed to read mesh from file %s.\n", inf);
    }
    vec3 *fbuf = (vec3 *)malloc(width * height * sizeof(vec3));
    render(tm, fbuf, 50.0393, width, height);
    if (!vec3_array_to_ppm(fbuf, outf, width, height)) {
        error("Failed to save to '%s'.", outf);
    }
    free(fbuf);


    tm_free(tm);
    return 0;
}
