#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"

#include "isect/isect.h"
#include "linalg/linalg.h"
#include "triangle_mesh.h"
#include "loaders.h"

static void
tm_intersect_precompute(triangle_mesh *me) {
#if ISECT_PC_P
    int bytes = ISECT_PC_N_ELS * sizeof(float) * me->n_tris;
    me->precomp = (float *)malloc(bytes);
    vec3 *it = me->verts;
    for (int i = 0; i < me->n_tris; i++) {
        vec3 v0 = *it++;
        vec3 v1 = *it++;
        vec3 v2 = *it++;
        float *addr = &me->precomp[i * ISECT_PC_N_ELS];
#if ISECT_METHOD == ISECT_BW12 || ISECT_METHOD == ISECT_BW12_B
        isect_bw12_pre(v0, v1, v2, addr);
#elif ISECT_METHOD == ISECT_BW9 || ISECT_METHOD == ISECT_BW9_B
        isect_bw9_pre(v0, v1, v2, addr);
#elif ISECT_METHOD == ISECT_SHEV
        isect_shev_pre(v0, v1, v2, addr);
#endif
    }
#endif
}

void
tm_free(triangle_mesh *me) {
    free(me->normals);
    free(me->coords);
    free(me->verts);
#if ISECT_PC_P
    free(me->precomp);
#endif
    free(me);
}

triangle_mesh *
tm_from_file(const char *fname, float scale, vec3 translate) {
    triangle_mesh *me = (triangle_mesh *)malloc(sizeof(triangle_mesh));

    int n_verts;
    int* indices;
    vec3* verts;

    if (!load_any_file(fname,
                       &me->n_tris, &indices,
                       &n_verts, &verts,
                       &me->normals, &me->coords)) {
        free(me);
        return NULL;
    }
    for (int i = 0; i < n_verts; i++) {
        vec3 v = verts[i];
        v = v3_add(v3_scale(v, scale), translate);
        verts[i] = v;
    }
    // "Unpack" indexed vertices.
    me->verts = (vec3 *)malloc(sizeof(vec3) * me->n_tris * 3);
    for (int i = 0; i < me->n_tris * 3; i++) {
        me->verts[i] = verts[indices[i]];
    }
    free(indices);
    free(verts);

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
        *tex_coords = v2_scale(ri->uv, 10.0);
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
        vec3 v0 = me->verts[t0];
        vec3 v1 = me->verts[t1];
        vec3 v2 = me->verts[t2];
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
#define BODY(N, I, O)                                           \
    { if (ISECT_FUN(o, d, &t, &uv, T) && t < nearest) {         \
            nearest = t;                                        \
            ri->t = t;                                          \
            ri->uv = uv;                                        \
            ri->tri_idx = N*I+O;                                \
        }                                                       \
        T += ISECT_PC_N_ELS;                                    \
    }
    float *T = me->precomp;
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
    return nearest < FLT_MAX;
}
