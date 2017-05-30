#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <stdbool.h>

#define LINALG_EPSILON 1e-8

// A simple linear algebra library for C.
typedef struct _vec3 {
    float x, y, z;
} vec3;

inline vec3
v3_sub(vec3 l, vec3 r) {
    return (vec3){l.x - r.x, l.y - r.y, l.z - r.z};
}

inline vec3
v3_add(vec3 l, vec3 r) {
    return (vec3){l.x + r.x, l.y + r.y, l.z + r.z};
}

inline vec3
v3_cross(vec3 l, vec3 r) {
    vec3 ret = {
        l.y * r.z - l.z * r.y,
        l.z * r.x - l.x * r.z,
        l.x * r.y - l.y * r.x
    };
    return ret;
}

inline float
v3_dot(vec3 l, vec3 r) {
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

inline vec3
v3_normalize(vec3 in) {
    vec3 out = in;
    float norm = in.x * in.x + in.y * in.y + in.z * in.z;
    if (norm > 0) {
        float factor = 1.0f / sqrt(norm);
        out.x *= factor;
        out.y *= factor;
        out.z *= factor;
    }
    return out;
}

inline bool
v3_approx_eq(vec3 l, vec3 r) {
    return fabs(l.x - r.x) < LINALG_EPSILON &&
        fabs(l.y - r.y) < LINALG_EPSILON &&
        fabs(l.z - r.z) < LINALG_EPSILON;
}

inline float
to_rad(const float deg) {
    return deg * M_PI / 180;
}

inline float
to_deg(const float rad) {
    return rad * (180.0f / M_PI);
}

#endif
