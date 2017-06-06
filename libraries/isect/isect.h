#ifndef ISECT_H
#define ISECT_H

#include "datatypes/common.h"
#include "linalg/linalg.h"

#define ISECT_NEAR 0.0001f
#define ISECT_FAR 10000.0f

#define ISECT_PC12_SIZE sizeof(float) * 12
#define ISECT_PC9_SIZE sizeof(float) * 10

void isect_precomp12_pre(vec3 v0, vec3 v1, vec3 v2, float *T);
void isect_precomp9_pre(vec3 v0, vec3 v1, vec3 v2, float *T);

// This algorithm is taken from
// https://pdfs.semanticscholar.org/8fc1/5c74a9d7326591c6bc507f539e1b0473b280.pdf
// but doesn't work properly yet.
inline float
v3_sign_3d(vec3 p, vec3 q, vec3 r) {
    return v3_dot(p, v3_cross(q, r));
}

inline bool
isect_sf01(vec3 o, vec3 d,
           vec3 v0, vec3 v1, vec3 v2,
           float *t, vec2  *uv) {
    float w2 = v3_sign_3d(d, v3_sub(v1, o), v3_sub(v0, o));
    float w0 = v3_sign_3d(d, v3_sub(v2, o), v3_sub(v1, o));
    bool s2 = w2 >= 0.0f;
    bool s0 = w0 >= 0.0f;
    if (s2 != s0)
        return false;
    float w1 = v3_sign_3d(d, v3_sub(v0, o), v3_sub(v2, o));
    bool s1 = w1 >= 0.0f;
    if (s2 != s1)
        return false;
    uv->x = w1 / (w0 + w1 + w2);
    uv->y = w2 / (w0 + w1 + w2);
    vec3 n = v3_cross(v0, v1);
    *t = v3_dot(n, v3_sub(v0, o)) / v3_dot(n, d);
    return uv->x >= 0 && uv->y >= 0 && (uv->x + uv->y) <= 1;
}

inline bool
isect_mt(vec3 o, vec3 d, vec3 v0, vec3 v1, vec3 v2, float *t, vec2  *uv) {
    vec3 v0v1 = v3_sub(v1, v0);
    vec3 v0v2 = v3_sub(v2, v0);
    vec3 pvec = v3_cross(d, v0v2);
    float det = v3_dot(v0v1, pvec);
    if (fabs(det) < LINALG_EPSILON)
        return false;
    float inv_det = 1 / det;
    vec3 tvec = v3_sub(o, v0);
    uv->x = v3_dot(tvec, pvec) * inv_det;
    if (uv->x < 0 || uv->x > 1)
        return false;
    vec3 qvec = v3_cross(tvec, v0v1);
    uv->y = v3_dot(d, qvec) * inv_det;
    if (uv->y < 0 || uv->x + uv->y > 1)
        return false;
    *t = v3_dot(v0v2, qvec) * inv_det;
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;
}

inline bool
isect_precomp12(vec3 o, vec3 d,
                vec3 v0, vec3 v1, vec3 v2,
                float *t, vec2 *uv, float *T) {
    float t_o = T[8] * o.x + T[9] * o.y + T[10] * o.z + T[11];
    float t_d = T[8] * d.x + T[9] * d.y + T[10] * d.z;
    *t = -t_o / t_d;
    if  (*t < ISECT_NEAR || *t > ISECT_FAR)
        return false;
    vec3 wr = v3_add(o, v3_scale(d, *t));
    uv->x = T[0] * wr.x + T[1] * wr.y + T[2] * wr.z + T[3];
    uv->y = T[4] * wr.x + T[5] * wr.y + T[6] * wr.z + T[7];
    return uv->x >= 0 && uv->y >= 0 && (uv->x + uv->y) <= 1;
}

inline bool
isect_precomp9(vec3 o, vec3 d,
               vec3 v0, vec3 v1, vec3 v2,
               float *t, vec2 *uv,
               float *T) {
    if (T[9] == 1.0) {
        float t_o = o.x + T[6] * o.y + T[7] * o.z + T[8];
        float t_d = d.x + T[6] * d.y + T[7] * d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.y + T[1] * wr.z + T[2];
        uv->y = T[3] * wr.y + T[4] * wr.z + T[5];
    } else if (T[9] == 2.0) {
        float t_o = T[6] * o.x + o.y + T[7] * o.z + T[8];
        float t_d = T[6] * d.x + d.y + T[7] * d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.x + T[1] * wr.z + T[2];
        uv->y = T[3] * wr.x + T[4] * wr.z + T[5];
    } else {
        float t_o = o.x * T[6] + o.y * T[7] + o.z + T[8];
        float t_d = d.x * T[6] + d.y * T[7] + d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.x + T[1] * wr.y + T[2];
        uv->y = T[3] * wr.x + T[4] * wr.y + T[5];
    }
    return uv->x >= 0 && uv->y >= 0 && (uv->x + uv->y) <= 1;
}

#endif
