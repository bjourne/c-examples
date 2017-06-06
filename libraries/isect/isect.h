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

inline float
v3_sign_3d(vec3 p, vec3 q, vec3 r) {
    return v3_dot(p, v3_cross(q, r));
}

inline bool
isect_sf01(vec3 o, vec3 d,
           vec3 v0, vec3 v1, vec3 v2,
           float *t, vec2  *uv) {
    vec3 v0o = v3_sub(v0, o);
    vec3 v1o = v3_sub(v1, o);
    vec3 v2o = v3_sub(v2, o);
    float w2 = v3_sign_3d(d, v1o, v0o);
    float w0 = v3_sign_3d(d, v2o, v1o);
    bool s2 = w2 >= 0.0f;
    bool s0 = w0 >= 0.0f;
    if (s2 != s0)
        return false;
    float w1 = v3_sign_3d(d, v0o, v2o);
    bool s1 = w1 >= 0.0f;
    if (s2 != s1)
        return false;
    uv->x = w1 / (w0 + w1 + w2);
    uv->y = w2 / (w0 + w1 + w2);
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);
    *t = v3_dot(n, v0o) / v3_dot(n, d);
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;
}

inline bool
isect_mt(vec3 o, vec3 d, vec3 v0, vec3 v1, vec3 v2, float *t, vec2  *uv) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 pvec = v3_cross(d, e2);
    float det = v3_dot(e1, pvec);
    if (fabs(det) < LINALG_EPSILON)
        return false;
    float inv_det = 1 / det;
    vec3 tvec = v3_sub(o, v0);
    uv->x = v3_dot(tvec, pvec) * inv_det;
    if (uv->x < 0 || uv->x > 1)
        return false;
    vec3 qvec = v3_cross(tvec, e1);
    uv->y = v3_dot(d, qvec) * inv_det;
    if (uv->y < 0 || uv->x + uv->y > 1)
        return false;
    *t = v3_dot(e2, qvec) * inv_det;
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;
}

inline bool
isect_mt_b(vec3 o, vec3 d,
          vec3 v0, vec3 v1, vec3 v2,
          float *t, vec2  *uv) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 pvec = v3_cross(d, e2);
    vec3 tvec = v3_sub(o, v0);
    vec3 qvec;
    uv->x = v3_dot(tvec, pvec);
    float det = v3_dot(e1, pvec);
    if (det > LINALG_EPSILON) {
        if (uv->x < 0 || uv->x > det)
            return false;
        qvec = v3_cross(tvec, e1);
        uv->y = v3_dot(d, qvec);
        if (uv->y < 0 || uv->x + uv->y > det)
            return false;
    } else if (det < -LINALG_EPSILON) {
        if (uv->x > 0 || uv->x < det)
            return false;
        qvec = v3_cross(tvec, e1);
        uv->y = v3_dot(d, qvec);
        if (uv->y > 0 || uv->x + uv->y < det)
            return false;
    } else {
        return false;
    }
    float inv_det = 1.0f / det;
    *t = v3_dot(e2, qvec) * inv_det;
    uv->x *= inv_det;
    uv->y *= inv_det;
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
