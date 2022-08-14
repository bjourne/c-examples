// Copyright (C) 2017-2019, 2022 Björn Lindqvist <bjourne@gmail.com>
#ifndef LINALG_H
#define LINALG_H

// Define needed to make msvc happy
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdbool.h>

// A simple linear algebra library for C.

// Approximations
#define LINALG_EPSILON 1e-8f

// Printing stuff
void
la_print_float(float f, int n_dec);

inline bool
approx_eq2(float x, float y, float epsilon) {
    return fabs(x - y) < epsilon;
}

inline bool
approx_eq(float x, float y) {
    return approx_eq2(x, y, LINALG_EPSILON);
}

// Trigonometry
inline float
to_rad(const float deg) {
    return (float)(deg * M_PI / 180.0);
}

inline float
to_deg(const float rad) {
    return (float)(rad * 180.0 / M_PI);
}

// vec2 type
typedef struct _vec2 {
    float x, y;
} vec2;

inline vec2
v2_scale(vec2 v, float f) {
    return (vec2){v.x * f, v.y * f};
}

inline vec2
v2_sub(vec2 l, vec2 r) {
    return (vec2){l.x - r.x, l.y - r.y};
}

inline vec2
v2_add(vec2 l, vec2 r) {
    return (vec2){l.x + r.x, l.y + r.y};
}

inline float
v2_dot(vec2 l, vec2 r) {
    return l.x * r.x + l.y * r.y;
}

inline float
v2_distance(vec2 v1, vec2 v2) {
    vec2 delta = v2_sub(v1, v2);
    return sqrtf(v2_dot(delta, delta));
}

void
v2_print(vec2 v, int n_dec);

inline bool
v2_approx_eq2(vec2 l, vec2 r, float epsilon) {
    return fabsf(l.x - r.x) < epsilon &&
        fabsf(l.y - r.y) < epsilon;
}

inline bool
v2_approx_eq(vec2 l, vec2 r) {
    return v2_approx_eq2(l, r, LINALG_EPSILON);
}

// vec3 type
typedef struct _vec3 {
    float x, y, z;
} vec3;

#define V3_GET(v, i) ((float *)&v)[i]

void
v3_print(vec3 v, int n_dec);

inline vec3
v3_sub(vec3 l, vec3 r) {
    return (vec3){l.x - r.x, l.y - r.y, l.z - r.z};
}

inline vec3
v3_add(vec3 l, vec3 r) {
    return (vec3){l.x + r.x, l.y + r.y, l.z + r.z};
}

inline vec3
v3_neg(vec3 v) {
    return (vec3){-v.x, -v.y, -v.z};
}

// I've noticed that when compiling with gcc 5.4.0 and the
// -march=native option, this function can return slightly different
// numbers depending on how it is inlined. Either it is a compiler
// bug, or a machine optimization.
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
    float norm = v3_dot(in, in);
    if (norm > 0) {
        float factor = 1 / sqrtf(norm);
        out.x *= factor;
        out.y *= factor;
        out.z *= factor;
    }
    return out;
}

inline bool
v3_approx_eq(vec3 l, vec3 r) {
    return fabsf(l.x - r.x) < LINALG_EPSILON &&
        fabsf(l.y - r.y) < LINALG_EPSILON &&
        fabsf(l.z - r.z) < LINALG_EPSILON;
}

inline vec3
v3_scale(vec3 v, float f) {
    return (vec3){v.x * f, v.y * f, v.z * f};
}

inline vec3
v3_from_scalar(float s) {
    return (vec3){s, s, s};
}

void
v3_get_plane(vec3 v0, vec3 v1, vec3 v2, vec3 *n, float *d);

// mat4 type
typedef struct _mat4 {
    float d[4][4];
} mat4;

// Most matrix functions doesn't need to be inline.
mat4 m4_identity();
void m4_print(mat4 m, int n_dec);
mat4 m4_inverse(mat4 m);
mat4 m4_look_at(vec3 position, vec3 at, vec3 up);
mat4 m4_perspective(float rad, float ar, float near, float far);
bool m4_approx_eq(mat4 l, mat4 r);
bool m4_approx_eq2(mat4 l, mat4 r, float epsilon);

// But some do
inline vec3
m4_mul_v3p(mat4 m, vec3 v) {
    float x = v.x, y = v.y, z = v.z;
    float a = x * m.d[0][0] + y * m.d[1][0] + z * m.d[2][0] + m.d[3][0];
    float b = x * m.d[0][1] + y * m.d[1][1] + z * m.d[2][1] + m.d[3][1];
    float c = x * m.d[0][2] + y * m.d[1][2] + z * m.d[2][2] + m.d[3][2];
    float w = x * m.d[0][3] + y * m.d[1][3] + z * m.d[2][3] + m.d[3][3];
    return (vec3){a / w, b / w, c / w};
}

inline vec3
m4_mul_v3d(mat4 m, vec3 v) {
    float x = v.x, y = v.y, z = v.z;
    float a = x * m.d[0][0] + y * m.d[1][0] + z * m.d[2][0];
    float b = x * m.d[0][1] + y * m.d[1][1] + z * m.d[2][1];
    float c = x * m.d[0][2] + y * m.d[1][2] + z * m.d[2][2];
    return (vec3){a, b, c};
}

inline mat4
m4_mul_m4(mat4 l, mat4 r) {
    mat4 ret;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            ret.d[i][j]=0;
            for(int k = 0; k < 4;k++){
                ret.d[i][j] += l.d[i][k] * l.d[k][j];
            }
        }
    }
    return ret;
}

// Operations on arbitrarily sized tensors.
void
convolve2d(float *src, int d1, int d2,
           float *kernel, int k1, int k2,
           float *dst,
           int stride, int padding);

#endif
