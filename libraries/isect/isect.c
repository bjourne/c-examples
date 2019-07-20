#include <string.h>
#include "isect.h"

extern inline bool
isect_mt(vec3 o, vec3 d,
         vec3 v0, vec3 v1, vec3 v2,
         float *t, vec2 *uv);
inline bool
isect_mt_c(vec3 o, vec3 d,
           vec3 v0, vec3 v1, vec3 v2,
           float *t, vec2 *uv);

extern inline bool
isect_bw9(vec3 o, vec3 d,
          float *t, vec2 *uv, isect_bw9_data *D);
extern inline bool
isect_bw9_b(vec3 o, vec3 d,
            float *t, vec2 *uv, isect_bw9_data *D);
extern inline bool
isect_bw12(vec3 o, vec3 d,
           float *t, vec2 *uv, isect_bw12_data *D);
extern inline bool
isect_bw12_b(vec3 o, vec3 d,
             float *t, vec2 *uv, isect_bw12_data *D);
extern inline bool
isect_sf01(vec3 o, vec3 d,
           vec3 v0, vec3 v1, vec3 v2,
           float *t, vec2 *uv);
extern inline bool
isect_ds(vec3 o, vec3 d,
         vec3 v0, vec3 v1, vec3 v2,
         float *t, vec2 *uv);

extern inline bool
isect_shev(vec3 o, vec3 d,
           float *t, vec2 *uv, isect_shev_data *D);

void
isect_shev_pre(vec3 v0, vec3 v1, vec3 v2, isect_shev_data *D) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);
    int u,v,w;
    if (fabsf(n.x) > fabsf(n.y) && fabsf(n.x) > fabsf(n.z)) {
        w = 0; u = 1; v = 2;
    } else if (fabsf(n.y) > fabsf(n.z)) {
        w = 1; u = 0; v = 2;
    } else {
        w = 2; u = 0; v = 1;
    }
    float sign = (w == 1) ? -1.0f : 1.0f;
    float nw = V3_GET(n, w);
    D->nu = V3_GET(n, u) / nw;
    D->nv = V3_GET(n, v) / nw;
    D->pu = V3_GET(v0, u);
    D->pv = V3_GET(v0, v);
    D->np = D->nu * D->pu + D->nv * D->pv + V3_GET(v0, w);
    D->e1u = sign * V3_GET(e1, u) / nw;
    D->e1v = sign * V3_GET(e1, v) / nw;
    D->e2u = sign * V3_GET(e2, u) / nw;
    D->e2v = sign * V3_GET(e2, v) / nw;
    if (w == 2) w = -1;
    D->ci = w;
}

void
isect_bw9_pre(vec3 v0, vec3 v1, vec3 v2, isect_bw9_data *D) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);
    float x1, x2;
    float num = v3_dot(v0, n);
    if (fabsf(n.x) > fabsf(n.y) && fabsf(n.x) > fabsf(n.z)) {
        x1 = v1.y * v0.z - v1.z * v0.y;
        x2 = v2.y * v0.z - v2.z * v0.y;
        memcpy(D, ((float[9]){
             e2.z / n.x, -e2.y / n.x,   x2 / n.x,
            -e1.z / n.x,  e1.y / n.x,  -x1 / n.x,
              n.y / n.x,   n.z / n.x, -num / n.x
        }), 36);
        D->ci = 0;
    } else if (fabsf(n.y) > fabsf(n.z)) {
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;
        memcpy(D, ((float[9]){
            -e2.z / n.y,  e2.x / n.y,   x2 / n.y,
             e1.z / n.y, -e1.x / n.y,  -x1 / n.y,
              n.x / n.y,   n.z / n.y, -num / n.y
        }), 36);
        D->ci = 1;
    } else {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;
        memcpy(D, ((float[9]){
             e2.y / n.z, -e2.x / n.z,   x2 / n.z,
            -e1.y / n.z,  e1.x / n.z,  -x1 / n.z,
              n.x / n.z,   n.y / n.z, -num / n.z
        }), 36);
        D->ci = -1;
    }
}

void
isect_bw12_pre(vec3 v0, vec3 v1, vec3 v2, isect_bw12_data *D) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);
    float x1, x2;
    float num = v3_dot(v0, n);
    if (fabsf(n.x) > fabsf(n.y) && fabsf(n.x) > fabsf(n.z)) {
        x1 = v1.y * v0.z - v1.z * v0.y;
        x2 = v2.y * v0.z - v2.z * v0.y;
        memcpy(D, ((float[12]){
            0,  e2.z / n.x, -e2.y / n.x,   x2 / n.x,
            0, -e1.z / n.x,  e1.y / n.x,  -x1 / n.x,
            1,   n.y / n.x,   n.z / n.x, -num / n.x
        }), 48);
    } else if (fabsf(n.y) > fabsf(n.z)) {
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;
        memcpy(D, ((float[12]){
            -e2.z / n.y, 0,  e2.x / n.y,   x2 / n.y,
             e1.z / n.y, 0, -e1.x / n.y,  -x1 / n.y,
              n.x / n.y, 1,   n.z / n.y, -num / n.y
        }), 48);
    } else {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;
        memcpy(D, ((float[12]){
            e2.y / n.z, -e2.x / n.z, 0,   x2 / n.z,
           -e1.y / n.z,  e1.x / n.z, 0,  -x1 / n.z,
             n.x / n.z,   n.y / n.z, 1, -num / n.z
        }), 48);
    }
}

void
isect_hh_pre(vec3 v0, vec3 v1, vec3 v2, isect_hh_data *D) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    D->n0 = v3_cross(e1, e2);
    D->d0 = v3_dot(D->n0, v0);

    float inv_denom = 1 / v3_dot(D->n0, D->n0);

    D->n1 = v3_scale(v3_cross(e2, D->n0), inv_denom);
    D->d1 = -v3_dot(D->n1, v0);

    D->n2 = v3_scale(v3_cross(D->n0, e1), inv_denom);
    D->d2 = -v3_dot(D->n2, v0);
}
