// -rwxrwxr-x 1 bjourne bjourne 51264 jun  6 03:20 build/rt<
#include <string.h>
#include "isect.h"

extern inline bool
isect_mt(vec3 o, vec3 d,
         vec3 v0, vec3 v1, vec3 v2,
         float *t, vec2 *uv);

extern inline bool
isect_precomp9(vec3 o, vec3 d,
               float *t, vec2 *uv, float *T);
extern inline bool
isect_precomp9_b(vec3 o, vec3 d,
               float *t, vec2 *uv, float *T);
extern inline bool
isect_precomp12(vec3 o, vec3 d,
                float *t, vec2 *uv, float *T);
extern inline bool
isect_precomp12_b(vec3 o, vec3 d,
                float *t, vec2 *uv, float *T);

void
isect_precomp9_pre(vec3 v0, vec3 v1, vec3 v2, float *T) {
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
              (float)1
            }, ISECT_PC9_SIZE);
    } else if (fabs(n.y) > fabs(n.z)) {
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;
        memcpy(T, (float[10]){
            -e2.z / n.y,  e2.x / n.y,   x2 / n.y,
             e1.z / n.y, -e1.x / n.y,  -x1 / n.y,
              n.x / n.y,   n.z / n.y, -num / n.y,
              (float)2
        }, ISECT_PC9_SIZE);
    } else if (fabs(n.z) > 0.0f) {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;
        memcpy(T, (float[10]){
             e2.y / n.z, -e2.x / n.z,   x2 / n.z,
            -e1.y / n.z,  e1.x / n.z,  -x1 / n.z,
              n.x / n.z,   n.y / n.z, -num / n.z,
              (float)3
        }, ISECT_PC9_SIZE);
    } else {
        error("Impossible!");
    }
}

void
isect_precomp12_pre(vec3 v0, vec3 v1, vec3 v2, float *T) {
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
        }, ISECT_PC12_SIZE);
    } else if (fabs(n.y) > fabs(n.z)) {
        x1 = v1.z * v0.x - v1.x * v0.z;
        x2 = v2.z * v0.x - v2.x * v0.z;
        memcpy(T, (float[12]){
            -e2.z / n.y, 0.0f,  e2.x / n.y,   x2 / n.y,
             e1.z / n.y, 0.0f, -e1.x / n.y,  -x1 / n.y,
              n.x / n.y, 1.0f,   n.z / n.y, -num / n.y
        }, ISECT_PC12_SIZE);
    } else if (fabs(n.z) > 0.0f) {
        x1 = v1.x * v0.y - v1.y * v0.x;
        x2 = v2.x * v0.y - v2.y * v0.x;
        memcpy(T, (float[12]){
            e2.y / n.z, -e2.x / n.z, 0.0f,   x2 / n.z,
           -e1.y / n.z,  e1.x / n.z, 0.0f,  -x1 / n.z,
             n.x / n.z,   n.y / n.z, 1.0f, -num / n.z
        }, ISECT_PC12_SIZE);
    } else {
        error("Impossible!");
    }
}
