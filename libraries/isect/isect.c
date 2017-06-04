#include "isect.h"

extern inline bool
isect_moeller_trumbore(vec3 orig, vec3 dir,
                       vec3 v0, vec3 v1, vec3 v2,
                       float *t, float *u, float *v);

extern inline bool
isect_precomp12(vec3 orig, vec3 dir,
                vec3 v0, vec3 v1, vec3 v2,
                float *t, float *u, float *v,
                float *T);

void
isect_precomp12_precompute(vec3 v0, vec3 v1, vec3 v2, float *T) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    vec3 n = v3_cross(e1, e2);

    float x1, x2;
    float num = v3_dot(v0, n);
    if (fabs(n.x) > fabs(n.y) && fabs(n.x) > fabs(n.z)) {
        // x is pivot
        x1 = v1.y * v0.z - v1.z * v0.y;
        x2 = v2.y * v0.z - v2.z * v0.y;

        T[0] = 0.0f;
        T[1] = e2.z / n.x;
        T[2] = -e2.y / n.x;
        T[3] = x2 / n.x;

        T[4] = 0.0f;
        T[5] = -e1.z / n.x;
        T[6] = e1.y / n.x;
        T[7] = -x1 / n.x;

        T[8] = 1.0f;
        T[9] = n.y / n.x;
        T[10] = n.z / n.x;
        T[11] = -num / n.x;
    } else if (fabs(n.y) > fabs(n.z)) {
        // y is pivot
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;

        T[0] = -e2.z / n.y;
        T[1] = 0.0f;
        T[2] = e2.x / n.y;
        T[3] = x2 / n.y;

        T[4] = e1.z / n.y;
        T[5] = 0.0f;
        T[6] = -e1.x / n.y;
        T[7] = -x1 / n.y;

        T[8] = n.x / n.y;
        T[9] = 1.0f;
        T[10] = n.z / n.y;
        T[11] = -num / n.y;
    } else if (fabs(n.z) > 0.0f) {
        // z is pivot
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;

        T[0] = e2.y / n.z;
        T[1] = -e2.x / n.z;
        T[2] = 0.0f;
        T[3] = x2 / n.z;

        T[4] = -e1.y / n.z;
        T[5] = e1.x / n.z;
        T[6] = 0.0f;
        T[7] = -x1 / n.z;

        T[8] = n.x / n.z;
        T[9] = n.y / n.z;
        T[10] = 1.0f;
        T[11] = -num / n.z;
    } else {
        error("Impossible!");
    }

}
