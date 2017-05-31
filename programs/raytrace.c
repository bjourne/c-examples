// This is a raytracer written in C.
// This code is verrrry much based on: www.scratchapixel.com
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "datatypes/common.h"
#include "linalg/linalg.h"

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
        int r = CLAMP(255.0f * arr[i].x, 0, 255);
        int g = CLAMP(255.0f * arr[i].y, 0, 255);
        int b = CLAMP(255.0f * arr[i].z, 0, 255);
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

typedef struct _ray_intersection {
    float dist;
    int tri_idx;
} ray_intersection;

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
        if (ray_tri_intersect(orig, dir,
                              v0, v1, v2,
                              &t, &u, &v) && t < nearest) {
            nearest = t;
            ri->dist = t;
            ri->tri_idx = i;
        }
    }
    return nearest < FLT_MAX;
}

typedef struct _raytrace_settings {
    mat4 view;
    float fov;
    int width;
    int height;
    char *mesh_file;
    char *image_file;
    vec3 bg_col;
} raytrace_settings;

raytrace_settings *
rt_from_args(int argc, char *argv[]) {
    if (argc != 9) {
        return NULL;
    }
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    if (width <= 0 || width > 2048 ||
        height <= 0 || height > 2048) {
        return NULL;
    }
    double fov_d = atof(argv[5]);
    if (fov_d <= 0.0 || fov_d >= 100.0) {
        return NULL;
    }

    double r = atof(argv[6]);
    double g = atof(argv[7]);
    double b = atof(argv[8]);
    if (r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1) {
        return NULL;
    }
    raytrace_settings *me = (raytrace_settings *)
        malloc(sizeof(raytrace_settings));
    me->width = width;
    me->height = height;
    me->mesh_file = strdup(argv[1]);
    me->image_file = strdup(argv[2]);
    me->fov = fov_d;
    me->bg_col.x = r;
    me->bg_col.y = g;
    me->bg_col.z = b;
    mat4 tmp = {
        {
            {0.707107, -0.331295, 0.624695, 0},
            {0, 0.883452, 0.468521, 0},
            {-0.707107, -0.331295, 0.624695, 0},
            {-1.63871, -5.747777, -40.400412, 1}
        }
    };
    me->view = m4_inverse(tmp);
    return me;
}

void
rt_free(raytrace_settings *me) {
    free(me->mesh_file);
    free(me->image_file);
    free(me);
}

vec3
rt_ray_direction(raytrace_settings *rt, int x, int y,
                 float aspect_ratio, float scale) {
    float ray_x = (2 * (x + 0.5) / (float)rt->width - 1)
        * aspect_ratio * scale;
    float ray_y = (1 - 2 * (y + 0.5) / (float)rt->height)
        * scale;
    vec3 dir = {ray_x, ray_y, -1};
    dir = m4_mul_v3d(rt->view, dir);
    dir = v3_normalize(dir);
    return dir;
}

vec3
shade_intersection(ray_intersection *ri, triangle_mesh *tm) {
    int i0 = tm->indices[3*ri->tri_idx];
    int i1 = tm->indices[3*ri->tri_idx + 1];
    int i2 = tm->indices[3*ri->tri_idx + 2];
    vec3 p0 = tm->positions[i0];
    vec3 p1 = tm->positions[i1];
    vec3 p2 = tm->positions[i2];
    vec3 e1 = v3_sub(p1, p0);
    vec3 e2 = v3_sub(p2, p0);
    vec3 n = v3_normalize(v3_cross(e1, e2));
    return n;
}

vec3
cast_ray(vec3 orig, vec3 dir, vec3 bg_col, triangle_mesh *tm) {
    ray_intersection ri;
    if (tm_intersect(tm, orig, dir, &ri)) {
        return shade_intersection(&ri, tm);
    }
    return bg_col;
}

void
render(raytrace_settings *rt, triangle_mesh *tm, vec3 *fbuf) {
    int w = rt->width;
    int h = rt->height;
    vec3 orig = m4_mul_v3p(rt->view, (vec3){0});
    float scale = tan(to_rad(rt->fov * 0.5));
    float aspect_ratio = (float)w / (float)h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            vec3 dir = rt_ray_direction(rt, x, y, aspect_ratio, scale);
            *fbuf = cast_ray(orig, dir, rt->bg_col, tm);
            fbuf++;
        }
    }
}

void
usage() {
    printf("usage: raytrace mesh-file image "
           "width[0-2048] height[0-2048] fov[0-100] "
           "bg_r[0-1] bg_g[0-1] bg_b[0-1]\n");
    exit(1);
}

int
main(int argc, char *argv[]) {
    raytrace_settings *rt = rt_from_args(argc, argv);
    if (!rt) {
        usage();
    }

    triangle_mesh *tm = tm_from_file(rt->mesh_file);
    if (!tm) {
        error("Failed to read mesh from file '%s'.\n", rt->mesh_file);
    }

    int w = rt->width;
    int h = rt->height;
    vec3 *fbuf = (vec3 *)malloc(w * h * sizeof(vec3));
    render(rt, tm, fbuf);
    if (!vec3_array_to_ppm(fbuf, rt->image_file, w, h)) {
        error("Failed to save to '%s'.", rt->image_file);
    }
    free(fbuf);
    tm_free(tm);
    rt_free(rt);
    return 0;
}
