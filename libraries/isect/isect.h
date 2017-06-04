#ifndef ISECT_H
#define ISECT_H

#include "datatypes/common.h"
#include "linalg/linalg.h"

#define ISECT_NEAR 0.0001f
#define ISECT_FAR 10000.0f

void isect_precomp12_precompute(vec3 v0, vec3 v1, vec3 v2, float *T);

inline bool
isect_moeller_trumbore(vec3 orig, vec3 dir,
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
    if (*t < ISECT_NEAR || *t > ISECT_FAR) {
        return false;
    }
    return true;
}

inline bool
isect_precomp12(vec3 orig, vec3 dir,
                vec3 v0, vec3 v1, vec3 v2,
                float *t, float *u, float *v,
                float *T) {
    float trans_o =
        T[8] * orig.x +
        T[9] * orig.y +
        T[10] * orig.z +
        T[11];
    float trans_d =
        T[8] * dir.x +
        T[9] * dir.y +
        T[10] * dir.z;
    float ta = -trans_o / trans_d;
    if  (ta < ISECT_NEAR || ta > ISECT_FAR) {
        return false;
    }
    vec3 wr = {
        orig.x + ta * dir.x,
        orig.y + ta * dir.y,
        orig.z + ta * dir.z
    };

    float xg = T[0] * wr.x + T[1] * wr.y + T[2] * wr.z + T[3];
    float yg = T[4] * wr.x + T[5] * wr.y + T[6] * wr.z + T[7];
    if (xg < 0 || yg < 0 || (yg + xg) > 1) {
        return false;
    }
    *t = ta;
    *u = xg;
    *v = yg;
    return true;
}


#endif
