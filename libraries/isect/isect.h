#ifndef ISECT_H
#define ISECT_H

#include "datatypes/common.h"
#include "linalg/linalg.h"

#define ISECT_NEAR 0.0001f
#define ISECT_FAR 10000.0f

void isect_precomp12_pre(vec3 v0, vec3 v1, vec3 v2, float *T);

inline bool
isect_moeller_trumbore(vec3 o, vec3 d,
                       vec3 v0, vec3 v1, vec3 v2,
                       float *t, float *u, float *v) {
    vec3 v0v1 = v3_sub(v1, v0);
    vec3 v0v2 = v3_sub(v2, v0);
    vec3 pvec = v3_cross(d, v0v2);
    float det = v3_dot(v0v1, pvec);
    if (fabs(det) < LINALG_EPSILON) {
        return false;
    }
    float inv_det = 1 / det;
    vec3 tvec = v3_sub(o, v0);

    *u = v3_dot(tvec, pvec) * inv_det;
    if (*u < 0 || *u > 1) {
        return false;
    }
    vec3 qvec = v3_cross(tvec, v0v1);
    *v = v3_dot(d, qvec) * inv_det;
    if (*v < 0 || *u + *v > 1) {
        return false;
    }
    *t = v3_dot(v0v2, qvec) * inv_det;
    return *t >= ISECT_NEAR && *t <= ISECT_FAR;
}

inline bool
isect_precomp12(vec3 o, vec3 d,
                vec3 v0, vec3 v1, vec3 v2,
                float *t, float *u, float *v,
                float *T) {
    float t_o = T[8] * o.x + T[9] * o.y + T[10] * o.z + T[11];
    float t_d = T[8] * d.x + T[9] * d.y + T[10] * d.z;
    *t = -t_o / t_d;
    if  (*t < ISECT_NEAR || *t > ISECT_FAR) {
        return false;
    }
    vec3 wr = {o.x + *t * d.x, o.y + *t * d.y, o.z + *t * d.z};
    *u = T[0] * wr.x + T[1] * wr.y + T[2] * wr.z + T[3];
    *v = T[4] * wr.x + T[5] * wr.y + T[6] * wr.z + T[7];
    return *u >= 0 && *v >= 0 && (*v + *u) <= 1;
}


#endif
