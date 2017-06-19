#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "file3d/file3d.h"
#include "isect/isect.h"
#include "linalg/linalg.h"
#include "triangle_mesh.h"

static void
tm_intersect_precompute(triangle_mesh *me) {
#if ISECT_PC_P
    me->precomp = (ISECT_DATA *)malloc(sizeof(ISECT_DATA) * me->n_tris);
    vec3 *it = me->verts;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = *it++;
        vec3 v1 = *it++;
        vec3 v2 = *it++;
        ISECT_DATA *addr = &me->precomp[i];
        ISECT_FUN_PRE(v0, v1, v2, addr);
    }
#endif
}

void
tm_free(triangle_mesh *me) {
    free(me->normals);
    if (me->coords) {
        free(me->coords);
    }
    free(me->verts);
#if ISECT_PC_P
    free(me->precomp);
#endif
    free(me);
}

triangle_mesh *
tm_from_file(char *fname, float scale, vec3 translate) {

    file3d *f3d = f3d_load(fname);
    if (f3d->error_code != FILE3D_ERR_NONE) {
        printf("Loading error: %s\n", f3d_get_error_string(f3d));
        return NULL;
    }
    // Scale and translate all vertices
    for (int i = 0; i < f3d->n_verts; i++) {
        vec3 v = f3d->verts[i];
        f3d->verts[i] = v3_add(v3_scale(v, scale), translate);
    }
    triangle_mesh *me = (triangle_mesh *)malloc(sizeof(triangle_mesh));

    // Unpack indexed vertices
    me->n_tris = f3d->n_tris;
    me->verts = (vec3 *)malloc(sizeof(vec3) * f3d->n_tris * 3);
    for (int i = 0; i < me->n_tris * 3; i++) {
        me->verts[i] = f3d->verts[f3d->vertex_indices[i]];
    }
    me->normals = (vec3 *)malloc(sizeof(vec3) * f3d->n_tris * 3);

    if (f3d->n_normals > 0) {
        // Unpack indexed normals
        for (int i = 0; i < me->n_tris * 3; i++) {
            int idx = f3d->normal_indices[i];
            me->normals[i] = f3d->normals[idx];
        }
    } else {
        // Fake normals
        for (int i = 0; i < me->n_tris; i++) {
            vec3 v0 = me->verts[3 * i];
            vec3 v1 = me->verts[3 * i + 1];
            vec3 v2 = me->verts[3 * i + 2];
            vec3 e1 = v3_sub(v1, v0);
            vec3 e2 = v3_sub(v2, v0);
            vec3 normal = v3_normalize(v3_cross(e1, e2));
            me->normals[3 * i] = normal;
            me->normals[3 * i + 1] = normal;
            me->normals[3 * i + 2] = normal;
        }
    }
    if (f3d->n_coords > 0) {
        me->coords = (vec2 *)malloc(sizeof(vec2) * f3d->n_tris * 3);
        for (int i = 0; i < me->n_tris * 3; i++) {
            int idx = f3d->coord_indices[i];
            me->coords[i] = f3d->coords[idx];
        }
    } else {
        me->coords = NULL;
    }
    f3d_free(f3d);
    tm_intersect_precompute(me);
    return me;
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
        *tex_coords = (vec2){0.5, 0.5};
        //*tex_coords = v2_scale(ri->uv, 0.01);
    }
    vec3 n0 = me->normals[t0];
    vec3 n1 = me->normals[t1];
    vec3 n2 = me->normals[t2];
    vec3 n0_scaled = v3_scale(n0, 1 - ri->uv.x - ri->uv.y);
    vec3 n1_scaled = v3_scale(n1, ri->uv.x);
    vec3 n2_scaled = v3_scale(n2, ri->uv.y);
    *normal = v3_add(v3_add(n0_scaled, n1_scaled), n2_scaled);
}

bool
tm_intersect(triangle_mesh *me, vec3 o, vec3 d, ray_intersection *ri) {
    float nearest = ISECT_FAR;
#if ISECT_RUNNING_MIN == false
    float t;
#endif
    vec2 uv;

#if ISECT_PC_P
#if ISECT_RUNNING_MIN == true
#define BODY(N, I, O)                            \
    { if (ISECT_FUN(o, d, &nearest, &uv, D)) {   \
            ri->t = nearest;                     \
            ri->uv = uv;                         \
            ri->tri_idx = N*I+O;                 \
        }                                        \
        D++;                                     \
    }
#else
#define BODY(N, I, O)                                           \
    { if (ISECT_FUN(o, d, &t, &uv, D) && t < nearest) {         \
            nearest = t;                                        \
            ri->t = t;                                          \
            ri->uv = uv;                                        \
            ri->tri_idx = N*I+O;                                \
        }                                                       \
        D++;                                                    \
    }
#endif
    ISECT_DATA *D = me->precomp;
#else
#if ISECT_RUNNING_MIN == true
#define BODY(N, I, O)                                     \
    { vec3 v0 = *it++, v1 = *it++, v2 = *it++;            \
        if (ISECT_FUN(o, d, v0, v1, v2, &nearest, &uv)) { \
            ri->t = nearest;                              \
            ri->uv = uv;                                  \
            ri->tri_idx = N*I+O;                          \
      }                                                   \
    }
#else
#define BODY(N, I, O)                                                   \
    { vec3 v0 = *it++, v1 = *it++, v2 = *it++;                          \
      if (ISECT_FUN(o, d, v0, v1, v2, &t, &uv) && t < nearest) {        \
          nearest = t;                                                  \
          ri->t = t;                                                    \
          ri->uv = uv;                                                  \
          ri->tri_idx = N*I+O;                                          \
      }                                                                 \
    }
#endif
    vec3 *it = me->verts;
#endif
    int n_loops = me->n_tris / 8;
    int n_remain = me->n_tris % 8;
    for (int i = 0; i < n_loops; i++) {
        BODY(8, i, 0);
        BODY(8, i, 1);
        BODY(8, i, 2);
        BODY(8, i, 3);
        BODY(8, i, 4);
        BODY(8, i, 5);
        BODY(8, i, 6);
        BODY(8, i, 7);
    }
    for (int i = 0; i < n_remain; i++) {
        BODY(8, n_loops, i);
    }
    return nearest < ISECT_FAR;
}
