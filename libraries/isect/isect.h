#ifndef ISECT_H
#define ISECT_H

#include "datatypes/common.h"
#include "linalg/linalg.h"

#define ISECT_NEAR 0.0001f
#define ISECT_FAR 10000.0f

typedef union {
    int i;
    float f;
} int_or_float;

typedef struct {
    vec3 n0; float d0;
    vec3 n1; float d1;
    vec3 n2; float d2;
} isect_bw12_data;

typedef struct {
    float a, b, c, d, e, f, g, h, i;
    int ci;
} isect_bw9_data;

typedef struct {
    float nu, nv;
    float np;
    float pu, pv;
    float e1u, e1v;
    float e2u, e2v;
    int ci;
} isect_shev_data;

typedef struct {
    vec3 n0; float d0;
    vec3 n1; float d1;
    vec3 n2; float d2;
} isect_hh_data;

void isect_bw9_pre(vec3 v0, vec3 v1, vec3 v2, isect_bw9_data *D);
void isect_bw12_pre(vec3 v0, vec3 v1, vec3 v2, isect_bw12_data *D);
void isect_shev_pre(vec3 v0, vec3 v1, vec3 v2, isect_shev_data *D);
void isect_hh_pre(vec3 v0, vec3 v1, vec3 v2, isect_hh_data *D);

// Only sets t on intersection.
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
    if (b == 0)
        return false;
    float new_t = a / b;
    if  (new_t < ISECT_NEAR || new_t > *t)
        return false;
    vec3 i = v3_add(o, v3_scale(d, new_t));
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
    *t = new_t;
    return true;
}

inline float
v3_sign_3d(vec3 p, vec3 q, vec3 r) {
    return v3_dot(p, v3_cross(q, r));
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
    bool s2 = w2 >= 0;
    bool s0 = w0 >= 0;
    if (s2 != s0)
        return false;
    float w1 = v3_sign_3d(d, v0o, v2o);
    bool s1 = w1 >= 0;
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
    if (det == 0)
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
    if (det > 0) {
        if (uv->x < 0 || uv->x > det)
            return false;
        qvec = v3_cross(tvec, e1);
        uv->y = v3_dot(d, qvec);
        if (uv->y < 0 || uv->x + uv->y > det)
            return false;
    } else if (det < 0) {
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

// Only sets t on intersection.
inline bool
isect_bw12(vec3 o, vec3 d, float *t, vec2 *uv, isect_bw12_data *D) {
    float t_o = v3_dot(o, D->n2) + D->d2;
    float t_d = v3_dot(d, D->n2);
    float new_t = -t_o / t_d;
    if  (new_t < ISECT_NEAR || new_t > *t)
        return false;
    vec3 wr = v3_add(o, v3_scale(d, new_t));
    uv->x = v3_dot(wr, D->n0) + D->d0;
    uv->y = v3_dot(wr, D->n1) + D->d1;
    if (uv->x < 0 || uv->y < 0 || (uv->x + uv->y) > 1)
        return false;
    *t = new_t;
    return true;
}

// Only sets t on intersection.
inline bool
isect_bw12_b(vec3 o, vec3 d, float *t, vec2 *uv, isect_bw12_data *D) {
    float t_o = v3_dot(o, D->n2) + D->d2;
    float t_d = v3_dot(d, D->n2);
    float new_t = -t_o / t_d;
    if  (new_t < ISECT_NEAR || new_t > *t)
        return false;
    vec3 wr = v3_add(o, v3_scale(d, new_t));
    uv->x = v3_dot(wr, D->n0) + D->d0;
    if (uv->x < 0 || uv->x > 1)
        return false;
    uv->y = v3_dot(wr, D->n1) + D->d1;
    if (uv->y < 0 || (uv->x + uv->y) > 1)
        return false;
    *t = new_t;
    return true;
}

// Only sets t on intersection.
inline bool
isect_bw9(vec3 o, vec3 d, float *t, vec2 *uv, isect_bw9_data *D) {
    float new_t;
    switch (D->ci) {
    case 0: {
        float t_o = o.x + D->g * o.y + D->h * o.z + D->i;
        float t_d = d.x + D->g * d.y + D->h * d.z;
        new_t = -t_o / t_d;
        if  (new_t < ISECT_NEAR || new_t > *t)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, new_t));
        uv->x = D->a * wr.y + D->b * wr.z + D->c;
        uv->y = D->d * wr.y + D->e * wr.z + D->f;
        break;
    }
    case 1: {
        float t_o = D->g * o.x + o.y + D->h * o.z + D->i;
        float t_d = D->g * d.x + d.y + D->h * d.z;
        new_t = -t_o / t_d;
        if  (new_t < ISECT_NEAR || new_t > *t)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, new_t));
        uv->x = D->a * wr.x + D->b * wr.z + D->c;
        uv->y = D->d * wr.x + D->e * wr.z + D->f;
        break;
    }
    default: {
        float t_o = o.x * D->g + o.y * D->h + o.z + D->i;
        float t_d = d.x * D->g + d.y * D->h + d.z;
        new_t = -t_o / t_d;
        if  (new_t < ISECT_NEAR || new_t > *t)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, new_t));
        uv->x = D->a * wr.x + D->b * wr.y + D->c;
        uv->y = D->d * wr.x + D->e * wr.y + D->f;
    }
    }
    if (uv->x < 0 || uv->y < 0 || (uv->x + uv->y) > 1)
        return false;
    *t = new_t;
    return true;
}


// Only sets t on intersection.
inline bool
isect_bw9_b_ending(vec2 *uv, float *t, float new_t) {
    if (uv->y < 0 || (uv->x + uv->y) > 1)
        return false;
    *t = new_t;
    return true;
}
inline bool
isect_bw9_b(vec3 o, vec3 d, float *t, vec2 *uv, isect_bw9_data *D) {
    float new_t;
    switch (D->ci) {
    case 0: {
        float t_o = o.x + D->g * o.y + D->h * o.z + D->i;
        float t_d = d.x + D->g * d.y + D->h * d.z;
        new_t = -t_o / t_d;
        if  (new_t < ISECT_NEAR || new_t > *t)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, new_t));
        uv->x = D->a * wr.y + D->b * wr.z + D->c;
        if (uv->x < 0 || uv->x > 1)
            return false;
        uv->y = D->d * wr.y + D->e * wr.z + D->f;
        return isect_bw9_b_ending(uv, t, new_t);
    }
    case 1: {
        float t_o = D->g * o.x + o.y + D->h * o.z + D->i;
        float t_d = D->g * d.x + d.y + D->h * d.z;
        new_t = -t_o / t_d;
        if  (new_t < ISECT_NEAR || new_t > *t)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, new_t));
        uv->x = D->a * wr.x + D->b * wr.z + D->c;
        if (uv->x < 0 || uv->x > 1)
            return false;
        uv->y = D->d * wr.x + D->e * wr.z + D->f;
        return isect_bw9_b_ending(uv, t, new_t);
    }
    default: {
        float t_o = o.x * D->g + o.y * D->h + o.z + D->i;
        float t_d = d.x * D->g + d.y * D->h + d.z;
        new_t = -t_o / t_d;
        if  (new_t < ISECT_NEAR || new_t > *t)
            return false;
        vec3 wr = v3_add(o, v3_scale(d, new_t));
        uv->x = D->a * wr.x + D->b * wr.y + D->c;
        if (uv->x < 0 || uv->x > 1)
            return false;
        uv->y = D->d * wr.x + D->e * wr.y + D->f;
        return isect_bw9_b_ending(uv, t, new_t);
    }
    }
}

inline bool
isect_shev_ending(float det, float dett,
                  float Du, float Dv,
                  float *t, vec2 *uv, isect_shev_data *D) {
    uv->x = D->e2v * Du - D->e2u * Dv;
    uv->y = D->e1u * Dv - D->e1v * Du;
    float tmpdet0 = det - uv->x - uv->y;
    int pdet0 = ((int_or_float)tmpdet0).i;
    int pdetu = ((int_or_float)uv->x).i;
    int pdetv = ((int_or_float)uv->y).i;
    pdet0 = pdet0 ^ pdetu;
    pdet0 = pdet0 | (pdetu ^ pdetv);
    if (pdet0 & 0x80000000)
        return false;
    float rdet = 1 / det;

    *t = dett * rdet;
    uv->x *= rdet;
    uv->y *= rdet;
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;
}

inline bool
isect_shev(vec3 o, vec3 d,
           float *t, vec2 *uv, isect_shev_data *D) {
    float dett, det, Du, Dv;
    switch (D->ci) {
    case 0: {
        dett = D->np - (o.y * D->nu + o.z * D->nv + o.x);
        det = d.y * D->nu + d.z * D->nv + d.x;
        Du = d.y * dett - (D->pu - o.y) * det;
        Dv = d.z * dett - (D->pv - o.z) * det;
        return isect_shev_ending(det, dett, Du, Dv, t, uv, D);
    }
    case 1: {
        dett = D->np - (o.x * D->nu + o.z * D->nv + o.y);
        det = d.x * D->nu + d.z * D->nv + d.y;
        Du = d.x * dett - (D->pu - o.x) * det;
        Dv = d.z * dett - (D->pv - o.z) * det;
        return isect_shev_ending(det, dett, Du, Dv, t, uv, D);
    }
    default:
        dett = D->np - (o.x * D->nu + o.y * D->nv + o.z);
        det = d.x * D->nu + d.y * D->nv + d.z;
        Du = d.x * dett - (D->pu - o.x) * det;
        Dv = d.y * dett - (D->pv - o.y) * det;
        return isect_shev_ending(det, dett, Du, Dv, t, uv, D);
    }
}

inline bool
isect_hh(vec3 o, vec3 d, float *t, vec2 *uv, isect_hh_data *D) {
    float det = v3_dot(D->n0, d);
    float dett = D->d0 - v3_dot(o, D->n0);
    vec3 wr = v3_add(v3_scale(o, det), v3_scale(d, dett));
    uv->x = v3_dot(wr, D->n1) + det * D->d1;
    uv->y = v3_dot(wr, D->n2) + det * D->d2;
    float tmpdet0 = det - uv->x - uv->y;
    int pdet0 = ((int_or_float)tmpdet0).i;
    int pdetu = ((int_or_float)uv->x).i;
    int pdetv = ((int_or_float)uv->y).i;
    pdet0 = pdet0 ^ pdetu;
    pdet0 = pdet0 | (pdetu ^ pdetv);
    if (pdet0 & 0x80000000)
        return false;
    float rdet = 1 / det;
    uv->x *= rdet;
    uv->y *= rdet;
    *t = dett * rdet;
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;
}


#endif
