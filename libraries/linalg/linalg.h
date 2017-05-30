#ifndef LINALG_H
#define LINALG_H

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


#endif
