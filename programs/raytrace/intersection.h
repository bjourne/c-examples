#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "linalg/linalg.h"

inline bool
moeller_trumbore_isect(vec3 orig, vec3 dir,
                       vec3 v0, vec3 v1, vec3 v2,
                       float *t, float *u, float *v) {
    vec3 v0v1 = v3_sub(v1, v0);
    vec3 v0v2 = v3_sub(v2, v0);
    vec3 pvec = v3_cross(dir, v0v2);
    float det = v3_dot(v0v1, pvec);

    if (fabs(det) < LINALG_EPSILON) {
        return false;
    }
    float inv_det = 1 / det;
    vec3 tvec = v3_sub(orig, v0);

    *u = v3_dot(tvec, pvec) * inv_det;
    if (*u < 0 || *u > 1) {
        return false;
    }
    vec3 qvec = v3_cross(tvec, v0v1);
    *v = v3_dot(dir, qvec) * inv_det;
    if (*v < 0 || *u + *v > 1) {
        return false;
    }
    *t = v3_dot(v0v2, qvec) * inv_det;
    return true;
}

inline bool
precomp12_isect(vec3 orig, vec3 dir,
                vec3 v0, vec3 v1, vec3 v2,
                float *t, float *u, float *v,
                float *T) {
    float trans_s =
        T[8] * orig.x +
        T[9] * orig.y +
        T[10] * dir.z +
        T[11];
    float trans_d =
        T[8] * dir.x +
        T[9] * dir.y +
        T[10] * dir.z;
    float ta = -trans_s / trans_d;

    if (ta <= LINALG_EPSILON || ta >= 100000.0f) {
        return false;
    }
    vec3 wr = {
        orig.x + ta * dir.x,
        orig.y + ta * dir.y,
        orig.z + ta * dir.z
    };

    float xg = T[0] * wr.x + T[1] * wr.y + T[2] * wr.z + T[3];
    float yg = T[4] * wr.x + T[5] * wr.y + T[6] * wr.z + T[7];
    if (xg >= 0.0 && yg >= 0.0f && (yg + xg) < 1.0f) {
        *t = ta;
        *u = xg;
        *v = yg;
        return true;
    }
    return false;
}

#endif
