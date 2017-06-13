#ifndef ISECT_H
#define ISECT_H

#include "datatypes/common.h"
#include "linalg/linalg.h"

#define ISECT_NEAR 0.0001f
#define ISECT_FAR 10000.0f

#define ISECT_BW12_SIZE sizeof(float) * 12
#define ISECT_BW9_SIZE sizeof(float) * 10

typedef union { int i; float f; } u;

typedef struct {
    float nu;
    float nv;
    float np;
    float pu;
    float pv;
    float e1u;
    float e1v;
    float e2u;
    float e2v;
    int ci;
} shev_data;

void isect_bw9_pre(vec3 v0, vec3 v1, vec3 v2, float *T);
void isect_bw12_pre(vec3 v0, vec3 v1, vec3 v2, float *T);
void isect_shev_pre(vec3 v0, vec3 v1, vec3 v2, float *T);

inline float
v3_sign_3d(vec3 p, vec3 q, vec3 r) {
    return v3_dot(p, v3_cross(q, r));
}

inline bool
isect_ds(vec3 o, vec3 d,
         vec3 v0, vec3 v1, vec3 v2,
         float *t, vec2 *uv) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);
    vec3 v0o = v3_sub(v0, o);
    float a = v3_dot(n, v0o);
    float b = v3_dot(n, d);
    if (b == 0.0f)
        return false;
    *t = a / b;
    if (*t < ISECT_NEAR || *t > ISECT_FAR)
        return false;
    vec3 i = v3_add(o, v3_scale(d, *t));
    float u_u = v3_dot(e1, e1);
    float u_v = v3_dot(e1, e2);
    float v_v = v3_dot(e2, e2);
    vec3 w = v3_sub(i, v0);
    float w_u = v3_dot(w, e1);
    float w_v = v3_dot(w, e2);
    float det = u_v * u_v - u_u * v_v;
    uv->x = (u_v * w_v - v_v * w_u) / det;
    if (uv->x < 0 || uv->x > 1)
        return false;
    uv->y = (u_v * w_u - u_u * w_v) / det;
    if (uv->y < 0 || uv->x + uv->y > 1)
        return false;
    return true;
}

inline bool
isect_sf01(vec3 o, vec3 d,
           vec3 v0, vec3 v1, vec3 v2,
           float *t, vec2 *uv) {
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
isect_mt(vec3 o, vec3 d,
         vec3 v0, vec3 v1, vec3 v2,
         float *t, vec2 *uv) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 pvec = v3_cross(d, e2);
    float det = v3_dot(e1, pvec);
    if (det == 0.0f)
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
    if (det > 0.0f) {
        if (uv->x < 0 || uv->x > det)
            return false;
        qvec = v3_cross(tvec, e1);
        uv->y = v3_dot(d, qvec);
        if (uv->y < 0 || uv->x + uv->y > det)
            return false;
    } else if (det < 0.0f) {
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
isect_bw12(vec3 o, vec3 d,
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
isect_bw12_b(vec3 o, vec3 d,
                  float *t, vec2 *uv, float *T) {
    float t_o = T[8] * o.x + T[9] * o.y + T[10] * o.z + T[11];
    float t_d = T[8] * d.x + T[9] * d.y + T[10] * d.z;
    *t = -t_o / t_d;
    if  (*t < ISECT_NEAR || *t > ISECT_FAR)
        return false;
    vec3 wr = v3_add(o, v3_scale(d, *t));
    uv->x = T[0] * wr.x + T[1] * wr.y + T[2] * wr.z + T[3];
    if (uv->x < 0 || uv->x > 1)
        return false;
    uv->y = T[4] * wr.x + T[5] * wr.y + T[6] * wr.z + T[7];
    return uv->y >= 0 && (uv->x + uv->y) <= 1;
}

inline bool
isect_bw9(vec3 o, vec3 d,
               float *t, vec2 *uv, float *T) {
    if (((u)T[9]).i == 0) {
        float t_o = o.x + T[6] * o.y + T[7] * o.z + T[8];
        float t_d = d.x + T[6] * d.y + T[7] * d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.y + T[1] * wr.z + T[2];
        uv->y = T[3] * wr.y + T[4] * wr.z + T[5];
    } else if (((u)T[9]).i == 1) {
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

inline bool
isect_bw9_b(vec3 o, vec3 d,
            float *t, vec2 *uv, float *T) {
    if (((u)T[9]).i == 0) {
        float t_o = o.x + T[6] * o.y + T[7] * o.z + T[8];
        float t_d = d.x + T[6] * d.y + T[7] * d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.y + T[1] * wr.z + T[2];
        if (uv->x < 0 || uv->x > 1)
            return false;
        uv->y = T[3] * wr.y + T[4] * wr.z + T[5];
    } else if (((u)T[9]).i == 1) {
        float t_o = T[6] * o.x + o.y + T[7] * o.z + T[8];
        float t_d = T[6] * d.x + d.y + T[7] * d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.x + T[1] * wr.z + T[2];
        if (uv->x < 0 || uv->x > 1)
            return false;
        uv->y = T[3] * wr.x + T[4] * wr.z + T[5];
    } else {
        float t_o = o.x * T[6] + o.y * T[7] + o.z + T[8];
        float t_d = d.x * T[6] + d.y * T[7] + d.z;
        *t = -t_o / t_d;
        if  (*t < ISECT_NEAR || *t > ISECT_FAR)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, *t));
        uv->x = T[0] * wr.x + T[1] * wr.y + T[2];
        if (uv->x < 0 || uv->x > 1)
            return false;
        uv->y = T[3] * wr.x + T[4] * wr.y + T[5];
    }
    return uv->y >= 0 && (uv->x + uv->y) <= 1;
}

#define ISECT_SHEV_ENDING                       \
    float detu = D->e2v * Du - D->e2u * Dv;     \
    float detv = D->e1u * Dv - D->e1v * Du;     \
    float tmpdet0 = det - detu - detv;          \
    int pdet0 = ((u)tmpdet0).i;                 \
    int pdetu = ((u)detu).i;                    \
    int pdetv = ((u)detv).i;                    \
    pdet0 = pdet0 ^ pdetu;                      \
    pdet0 = pdet0 | (pdetu ^ pdetv);            \
    if (pdet0 & 0x80000000)                     \
        return false;                           \
    float rdet = 1 / det;                       \
    *t = dett * rdet;                           \
    uv->x = detu * rdet;                        \
    uv->y = detv * rdet;                        \
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;

inline bool
isect_shev(vec3 o, vec3 d, float *t, vec2 *uv, float *T) {
    shev_data *D = (shev_data *)T;
    float dett, det, Du, Dv;
    if (((u)T[9]).i == 0) {
        dett = D->np - (o.y * D->nu + o.z * D->nv + o.x);
        det = d.y * D->nu + d.z * D->nv + d.x;
        Du = d.y*dett - (D->pu - o.y) * det;
        Dv = d.z*dett - (D->pv - o.z) * det;
        ISECT_SHEV_ENDING;
    } else if (((u)T[9]).i == 1) {
        dett = D->np - (o.x * D->nu + o.z * D->nv + o.y);
        det = d.x * D->nu + d.z * D->nv + d.y;
        Du = d.x * dett - (D->pu - o.x) * det;
        Dv = d.z * dett - (D->pv - o.z) * det;
        ISECT_SHEV_ENDING;
    } else {
        dett = D->np - (o.x * D->nu + o.y * D->nv + o.z);
        det = d.x * D->nu + d.y * D->nv + d.z;
        Du = d.x * dett - (D->pu - o.x) * det;
        Dv = d.y * dett - (D->pv - o.y) * det;
        ISECT_SHEV_ENDING;
    }
}



#endif
