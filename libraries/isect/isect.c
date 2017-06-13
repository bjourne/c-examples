// -rwxrwxr-x 1 bjourne bjourne 51264 jun  6 03:20 build/rt<
#include <string.h>
#include "isect.h"

extern inline bool
isect_mt(vec3 o, vec3 d,
         vec3 v0, vec3 v1, vec3 v2,
         float *t, vec2 *uv);

extern inline bool
isect_bw9(vec3 o, vec3 d,
          float *t, vec2 *uv, float *T);
extern inline bool
isect_bw9_b(vec3 o, vec3 d,
            float *t, vec2 *uv, float *T);
extern inline bool
isect_bw12(vec3 o, vec3 d,
           float *t, vec2 *uv, float *T);
extern inline bool
isect_bw12_b(vec3 o, vec3 d,
             float *t, vec2 *uv, float *T);
extern inline bool
isect_sf01(vec3 o, vec3 d,
           vec3 v0, vec3 v1, vec3 v2,
           float *t, vec2 *uv);

// Storage layout:
//
//  0 = nu
//  1 = nv
//  2 = np
//  3 = pu
//  4 = pv
//  5 = e1u
//  6 = e1v
//  7 = e2u
//  8 = e2v
//  9 = ci

void
isect_shev_pre(vec3 v0, vec3 v1, vec3 v2, float *T) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);
    int u,v,w;
    if (abs(n.x) < abs(n.y)) {
        if (abs(n.z) < abs(n.x)) {
            u = 0; v = 2; w = 1;
        } else {
            if (abs(n.z) < abs(n.y)) {
                u = 0; v = 2; w = 1;
            } else {
                u = 0; v = 1; w = 2;
            }
        }
    } else {
        if (abs(n.z) < abs(n.y)) {
            u = 1; v = 2; w = 0;
        } else {
            if (abs(n.z) < abs(n.x)) {
                u = 1; v = 2; w = 0;
            }
            else {
                u = 0; v = 1; w = 2;
            }
        }
    }
    float sign = 1.0f;
    for(int i=0; i<w; ++i)
        sign *= -1.0f;
    float nw = V3_GET(n, w);
    T[0] = V3_GET(n, u) / nw;
    T[1] = V3_GET(n, v) / nw;
    T[3] = V3_GET(v0, u);
    T[4] = V3_GET(v0, v);
    T[2] = T[0] * T[3] + T[1] * T[4] + V3_GET(v0, w);
    T[5] = sign * V3_GET(e1, u) / nw;
    T[6] = sign * V3_GET(e1, v) / nw;
    T[7] = sign * V3_GET(e2, u) / nw;
    T[8] = sign * V3_GET(e2, v) / nw;
    T[9] = ((int_or_float)w).f;
}

#define ISECT_BW12_SIZE sizeof(float) * 12
#define ISECT_BW9_SIZE sizeof(float) * 10

void
isect_bw9_pre(vec3 v0, vec3 v1, vec3 v2, float *T) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);

    float x1, x2;
    float num = v3_dot(v0, n);
    if (fabs(n.x) > fabs(n.y) && fabs(n.x) > fabs(n.z)) {
        x1 = v1.y * v0.z - v1.z * v0.y;
        x2 = v2.y * v0.z - v2.z * v0.y;
        memcpy(T, (float[10]){
             e2.z / n.x, -e2.y / n.x,   x2 / n.x,
            -e1.z / n.x,  e1.y / n.x,  -x1 / n.x,
              n.y / n.x,   n.z / n.x, -num / n.x,
             ((int_or_float)0).f
            }, ISECT_BW9_SIZE);
    } else if (fabs(n.y) > fabs(n.z)) {
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;
        memcpy(T, (float[10]){
            -e2.z / n.y,  e2.x / n.y,   x2 / n.y,
             e1.z / n.y, -e1.x / n.y,  -x1 / n.y,
              n.x / n.y,   n.z / n.y, -num / n.y,
             ((int_or_float)1).f
        }, ISECT_BW9_SIZE);
    } else if (fabs(n.z) > 0.0f) {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;
        memcpy(T, (float[10]){
             e2.y / n.z, -e2.x / n.z,   x2 / n.z,
            -e1.y / n.z,  e1.x / n.z,  -x1 / n.z,
              n.x / n.z,   n.y / n.z, -num / n.z,
             ((int_or_float)2).f
        }, ISECT_BW9_SIZE);
    } else {
        error("Impossible!");
    }
}

void
isect_bw12_pre(vec3 v0, vec3 v1, vec3 v2, float *T) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);

    float x1, x2;
    float num = v3_dot(v0, n);
    if (fabs(n.x) > fabs(n.y) && fabs(n.x) > fabs(n.z)) {
        x1 = v1.y * v0.z - v1.z * v0.y;
        x2 = v2.y * v0.z - v2.z * v0.y;
        memcpy(T, (float[12]){
            0.0f,  e2.z / n.x, -e2.y / n.x,   x2 / n.x,
            0.0f, -e1.z / n.x,  e1.y / n.x,  -x1 / n.x,
            1.0f,   n.y / n.x,   n.z / n.x, -num / n.x
        }, ISECT_BW12_SIZE);
    } else if (fabs(n.y) > fabs(n.z)) {
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;
        memcpy(T, (float[12]){
            -e2.z / n.y, 0.0f,  e2.x / n.y,   x2 / n.y,
             e1.z / n.y, 0.0f, -e1.x / n.y,  -x1 / n.y,
              n.x / n.y, 1.0f,   n.z / n.y, -num / n.y
        }, ISECT_BW12_SIZE);
    } else if (fabs(n.z) > 0.0f) {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;
        memcpy(T, (float[12]){
            e2.y / n.z, -e2.x / n.z, 0.0f,   x2 / n.z,
           -e1.y / n.z,  e1.x / n.z, 0.0f,  -x1 / n.z,
             n.x / n.z,   n.y / n.z, 1.0f, -num / n.z
        }, ISECT_BW12_SIZE);
    } else {
        error("Impossible!");
    }
}
