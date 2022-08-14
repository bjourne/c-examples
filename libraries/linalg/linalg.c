// Copyright (C) 2017-2019, 2022 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdio.h>
#include "linalg.h"

// Utils
void
la_print_float(float f, int n_dec) {
    char buf[256];
    #ifdef _MSC_VER
    sprintf_s(buf, 256, "%%.%df", n_dec);
    #else
    sprintf(buf, "%%.%df", n_dec);
    #endif
    printf(buf, f);
}

// Approximations
extern inline bool approx_eq2(float x, float y, float epsilon);
extern inline bool approx_eq(float x, float y);

// Trigonometry
extern inline float to_rad(const float deg);
extern inline float to_deg(const float rad);

// vec2 type
extern inline vec2 v2_add(vec2 l, vec2 r);
extern inline vec2 v2_scale(vec2 v, float f);
extern inline float v2_distance(vec2 v1, vec2 v2);

void
v2_print(vec2 v, int n_dec) {
    printf("{");
    la_print_float(v.x, n_dec);
    printf(", ");
    la_print_float(v.y, n_dec);
    printf("}");
}

// vec3 type
extern inline vec3 v3_add(vec3 l, vec3 r);
extern inline vec3 v3_sub(vec3 l, vec3 r);
extern inline vec3 v3_neg(vec3 v);
extern inline vec3 v3_cross(vec3 l, vec3 r);
extern inline float v3_dot(vec3 l, vec3 r);
extern inline vec3 v3_normalize(vec3 in);
extern inline bool v3_approx_eq(vec3 l, vec3 r);
extern inline vec3 v3_scale(vec3 v, float f);
extern inline vec3 v3_from_scalar(float s);

void
v3_print(vec3 v, int n_dec) {
    printf("{");
    la_print_float(v.x, n_dec);
    printf(", ");
    la_print_float(v.y, n_dec);
    printf(", ");
    la_print_float(v.z, n_dec);
    printf("}");
}

void
v3_get_plane(vec3 v0, vec3 v1, vec3 v2,
             vec3 *n, float *d) {
    vec3 e1 = v3_sub(v1, v0);
    vec3 e2 = v3_sub(v2, v0);
    *n = v3_cross(e1, e2);
    *d = -v3_dot(*n, v0);
}


// mat4 type
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
            la_print_float(m.d[y][x], n_dec);
            printf(" ");
        }
        la_print_float(m.d[y][3], n_dec);
        printf(", ");
    }
    for (int x = 0; x < 3; x++) {
        la_print_float(m.d[3][x], n_dec);
        printf(" ");
    }
    la_print_float(m.d[3][3], n_dec);
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
    float tan_half = (float)tan(rad / 2.0f);
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

void
convolve2d(float *src, int src_c, int src_h, int src_w,
           float *kernel,
           int kernel_c_out, int kernel_c_in,
           int kernel_h, int kernel_w,
           float *dst,
           int stride, int padding) {

    assert (src_c == kernel_c_in);

    int h_start = -padding;
    int h_end = src_h + padding - kernel_h + 1;
    int w_start = -padding;
    int w_end = src_w + padding - kernel_w + 1;

    int dst_h = (src_h + 2 * padding - kernel_h) / stride + 1;
    int dst_w = (src_w + 2 * padding - kernel_w) / stride + 1;
    int dst_size = dst_h * dst_w;

    int src_size = src_h * src_w;
    int kernel_size = kernel_w * kernel_h;
    for (int c_out = 0; c_out < kernel_c_out; c_out++) {
        float *kernel_ptr = &kernel[c_out * kernel_c_in * kernel_size];
        for (int c_in = 0; c_in < kernel_c_in; c_in++) {
            float *dst_ptr = &dst[c_out * dst_size];
            float *src_ptr = &src[c_in * src_size];
            for (int h = h_start; h < h_end; h += stride) {
                for (int w = w_start; w < w_end; w += stride) {
                    float acc = 0;
                    if (c_in > 0) {
                        acc = *dst_ptr;
                    }
                    float *kernel_ptr2 = &kernel_ptr[c_in * kernel_size];
                    for  (int i3 = 0; i3 < kernel_h; i3++) {
                        for (int i4 = 0; i4 < kernel_w; i4++)  {
                            int at1 = h + i3;
                            int at2 = w + i4;

                            float s = 0;
                            if (at1 >= 0 && at1 < src_h &&
                                at2 >= 0 && at2 < src_w) {
                                s = src_ptr[at1 * src_w + at2];
                            }
                            float weight = *kernel_ptr2;
                            acc += s * weight;
                            kernel_ptr2++;
                        }
                    }
                    //printf("%d %d %d = %.2f\n", c_out, h, w, acc);
                    *dst_ptr = acc;
                    dst_ptr++;
                }
            }
        }
    }
}
