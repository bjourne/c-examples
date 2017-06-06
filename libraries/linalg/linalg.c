#include <stdio.h>
#include "linalg.h"

extern inline vec3 v3_add(vec3 l, vec3 r);
extern inline vec3 v3_sub(vec3 l, vec3 r);

inline vec2 v2_add(vec2 l, vec2 r);

inline vec3 v3_neg(vec3 v);

extern inline vec3 v3_cross(vec3 l, vec3 r);
extern inline float v3_dot(vec3 l, vec3 r);
extern inline vec3 v3_normalize(vec3 in);

extern inline bool v3_approx_eq(vec3 l, vec3 r);

extern inline float to_rad(const float deg);
extern inline float to_deg(const float rad);

extern inline bool approx_eq2(float x, float y, float epsilon);
extern inline bool approx_eq(float x, float y);

extern inline vec3 v3_scale(vec3 v, float f);

inline vec2 v2_scale(vec2 v, float f);
inline vec3 v3_from_scalar(float s);

static void
print_float(float f, int n_dec) {
    char buf[256];
    sprintf(buf, "%%.%df", n_dec);
    printf(buf, f);
}

void
v3_print(vec3 v, int n_dec) {
    printf("{");
    print_float(v.x, n_dec);
    printf(", ");
    print_float(v.y, n_dec);
    printf(", ");
    print_float(v.z, n_dec);
    printf("}");
}

mat4
m4_identity() {
    return (mat4){
        {
            {1.0f, 0, 0, 0},
            {0, 1.0f, 0, 0},
            {0, 0, 1.0f, 0},
            {0, 0, 0, 1.0f}
        }
    };
}

void
m4_print(mat4 m, int n_dec) {
    printf("[");
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            print_float(m.d[y][x], n_dec);
            printf(" ");
        }
        print_float(m.d[y][3], n_dec);
        printf(", ");
    }
    for (int x = 0; x < 3; x++) {
        print_float(m.d[3][x], n_dec);
        printf(" ");
    }
    print_float(m.d[3][3], n_dec);
    printf("]");
}

mat4
m4_inverse(mat4 t) {
    mat4 s = m4_identity();

    // Forward elimination
    for (int i = 0; i < 3; i++) {
        int pivot = i;
        float pivot_size = t.d[i][i];
        if (pivot_size < 0) {
            pivot_size = -pivot_size;
        }
        for (int j = i + 1; j < 4; j++) {
            float tmp = t.d[j][i];
            if (tmp < 0) {
                tmp = -tmp;
            }
            if (tmp > pivot_size) {
                pivot = j;
                pivot_size = tmp;
            }
        }
        if (pivot_size == 0) {
            // Cannot invert singular matrix
            return m4_identity();
        }
        if (pivot != i) {
            for (int j = 0; j < 4; j++) {
                float tmp;
                tmp = t.d[i][j];
                t.d[i][j] = t.d[pivot][j];
                t.d[pivot][j] = tmp;

                tmp = s.d[i][j];
                s.d[i][j] = s.d[pivot][j];
                s.d[pivot][j] = tmp;
            }
        }
        for (int j = i + 1; j < 4; j++) {
            float f = t.d[j][i] / t.d[i][i];
            for (int k = 0; k < 4; k++) {
                t.d[j][k] -= f * t.d[i][k];
                s.d[j][k] -= f * s.d[i][k];
            }
        }
    }
    // Backward substitution
    for (int i = 3; i >= 0; --i) {
        float f;
        if ((f = t.d[i][i]) == 0) {
            // Cannot invert singular matrix
            return m4_identity();
        }
        for (int j = 0; j < 4; j++) {
            t.d[i][j] /= f;
            s.d[i][j] /= f;
        }
        for (int j = 0; j < i; j++) {
            f = t.d[j][i];
            for (int k = 0; k < 4; k++) {
                t.d[j][k] -= f * t.d[i][k];
                s.d[j][k] -= f * s.d[i][k];
            }
        }
    }
    return s;
}

mat4
m4_look_at(vec3 eye, vec3 center, vec3 up) {
    vec3 f = v3_normalize(v3_sub(center, eye));
    vec3 cf = v3_cross(f, up);
    vec3 s = v3_normalize(cf);

    vec3 u = v3_cross(s, f);
    mat4 ret;
    ret.d[0][0] = s.x;
    ret.d[1][0] = s.y;
    ret.d[2][0] = s.z;
    ret.d[0][1] = u.x;
    ret.d[1][1] = u.y;
    ret.d[2][1] = u.z;
    ret.d[0][2] = -f.x;
    ret.d[1][2] = -f.y;
    ret.d[2][2] = -f.z;
    ret.d[3][0] = -v3_dot(s, eye);
    ret.d[3][1] = -v3_dot(u, eye);
    ret.d[3][2] =  v3_dot(f, eye);
    ret.d[3][3] = 1.0f;
    return ret;
}

mat4
m4_perspective(float rad, float ar, float near, float far) {
    float tan_half = tan(rad / 2.0f);
    mat4 res = {
        {{0}}
    };
    res.d[0][0] = 1.0f / (ar * tan_half);
    res.d[1][1] = 1.0f / tan_half;
    res.d[2][3] = -1.0f;
    res.d[2][2] = -(far + near) / (far - near);
    res.d[3][2] = -(2.0f * far * near)/(far - near);
    return res;
}

bool
m4_approx_eq2(mat4 l, mat4 r, float epsilon) {
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            if (!approx_eq2(l.d[y][x], r.d[y][x], epsilon)) {
                return false;
            }
        }
    }
    return true;
}


bool
m4_approx_eq(mat4 l, mat4 r) {
    return m4_approx_eq2(l, r, LINALG_EPSILON);
}

extern inline vec3 m4_mul_v3p(mat4 m, vec3 v);
extern inline vec3 m4_mul_v3d(mat4 m, vec3 v);
extern inline mat4 m4_mul_m4(mat4 l, mat4 r);
