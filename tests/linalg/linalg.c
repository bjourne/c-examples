// Copyright (C) 2022 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdlib.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"

void
test_sub() {
    vec3 v1 = {10, 10, 10};
    vec3 v2 = {3, 3, 3};
    vec3 v3 = v3_sub(v1, v2);
    assert(v3.x == 7 && v3.y == 7 && v3.z == 7);
}

void
test_cross() {
    vec3 v1 = {1, 2, 3};
    vec3 v2 = {5, 5, 5};
    vec3 v3 = v3_cross(v1, v2);
    assert(v3.x == -5 && v3.y == 10 && v3.z == -5);
}

void
test_dot() {
    vec3 v1 = {1, 2, 3};
    vec3 v2 = {5, 5, 5};
    assert(v3_dot(v1, v2) == 30);
}

void
test_normalize() {
    vec3 v1 = v3_normalize((vec3){30, -20, 8});
    vec3 v2 = {0.81229556f, -0.54153037f, 0.21661215f};
    assert(v3_approx_eq(v1, v2));
}

void
test_inverse() {
    mat4 m1 = m4_identity();
    mat4 m2 = m4_inverse(m1);

    assert(approx_eq(m2.d[0][0], 1.0f) &&
           approx_eq(m2.d[1][1], 1.0f) &&
           approx_eq(m2.d[2][2], 1.0f) &&
           approx_eq(m2.d[3][3], 1.0f));

    mat4 m3 = {
        {
            {0.707107f, -0.331295f, 0.624695f, 0},
            {0, 0.883452f, 0.468521f, 0},
            {-0.707107f, -0.331295f, 0.624695f, 0},
            {-1.63871f, -5.747777f, -40.400412f, 1}
        }
    };
    mat4 m4 = m4_inverse(m3);
    mat4 m5 = {
        {
            { 0.70710695f,  0,          -0.70710534f,  0},
            {-0.33129495f,  0.88345194f, -0.33129495f,  0},
            { 0.62469512f,  0.46852198f,  0.62469512f,  0},
            {24.49247551f, 24.00636673f,  22.17499161f, 1}
        }
    };
    assert(m4_approx_eq2(m4, m5, 1e-5f));
}

void
test_mat_mul() {
    mat4 m1 = {
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        }
    };
    vec3 v1 = {1, 2, 3};
    vec3 v2 = m4_mul_v3p(m1, v1);
    assert(v3_approx_eq(v2, (vec3){0.70833331f, 0.80555558f, 0.90277779f}));
}

void
test_look_at() {
    vec3 eye = {1.0f, 2.0f, 3.0f};
    vec3 center = {4.0f, 5.0f, 6.0f};
    vec3 up = {7.0f, 8.0f, 9.0f};
    mat4 view = m4_look_at(eye, center, up);
    assert(approx_eq2(view.d[0][0], 0.408248276f, 1e-7f));
    assert(approx_eq2(view.d[3][3], 1.0f, 1e-7f));
}

void
test_to_rad() {
    assert(approx_eq(to_rad(70.0), 1.2217304706573486f));
}

void
test_to_deg() {
    assert(approx_eq(to_deg(to_rad(70.0)), 70.00000000));
}


void
test_perspective() {
    mat4 persp = m4_perspective(to_rad(70.0f),
                                500.0f/400.0f,
                                0.1f, 1000.0f);
    assert(approx_eq(persp.d[0][0], 1.142518401f));
    assert(persp.d[0][1] == 0.0f);
    assert(persp.d[0][2] == 0.0f);
    assert(persp.d[0][3] == 0.0f);
    assert(persp.d[3][3] == 0.0f);
}

void
test_get_plane() {
    vec3 v0 = {1, 2, 3};
    vec3 v1 = {3, 2, 1};
    vec3 v2 = {1, 0, 0};
    vec3 n;
    float d;
    v3_get_plane(v0, v1, v2, &n, &d);
    assert(v3_approx_eq(n, (vec3){-4, 6, -4}));
    assert(d == 4);
}

void
test_convolve_1() {
    int d1 = 5;
    int d2 = 5;
    float src[5][5] = {
        {2, 0, 4, 4, 1},
        {1, 3, 0, 1, 3},
        {2, 2, 3, 3, 2},
        {1, 2, 2, 2, 3},
        {2, 4, 3, 2, 1}
    };
    float kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    float expected[5][5] = {
        {6, 10, 12, 13,  9},
        {10, 17, 20, 21, 14},
        {11, 16, 18, 19, 14},
        {13, 21, 23, 21, 13},
        { 9, 14, 15, 13,  8}
    };
    float actual[d1][d2];

    tensor_convolve((float *)src, d1, d2,
                    (float *)kernel, 3, 3,
                    (float *)actual,
                    1);

    for (int i1 = 0; i1 < d1; i1++) {
        for (int i2 = 0; i2 < d2; i2++) {
            float v1 = actual[i1][i2];
            float v2 = expected[i1][i2];
            if (v1 != v2) {
                printf("Mismatch: %.2f != %.2f\n", v1, v2);
                assert(false);
            }
        }
    }
}

void
test_convolve_2() {
    int d1 = 5;
    int d2 = 5;
    float src[5][5] = {
        {0, 1, 3, 2, 2},
        {3, 3, 4, 0, 4},
        {4, 1, 2, 1, 2},
        {0, 0, 0, 0, 2},
        {0, 2, 2, 4, 1}
    };
    float kernel[3][3] = {
        {1, 1, 1},
        {1, -0, 1},
        {1, 1, 1}
    };
    float expected[5][5] = {
        {7, 13, 10, 13,  6},
        { 9, 18, 13, 20,  7},
        { 7, 16,  9, 14,  7},
        { 7, 11, 12, 14,  8},
        { 2,  2,  6,  5,  6}
    };
    float actual[d1][d2];
    tensor_convolve((float *)src, d1, d2,
                    (float *)kernel, 3, 3,
                    (float *)actual,
                    1);
    for (int i1 = 0; i1 < d1; i1++) {
        for (int i2 = 0; i2 < d2; i2++) {
            float v1 = actual[i1][i2];
            float v2 = expected[i1][i2];
            if (v1 != v2) {
                printf("Mismatch: %.2f != %.2f\n", v1, v2);
                assert(false);
            }
        }
    }
}

void
test_convolve_3() {
    int d1 = 10;
    int d2 = 3;
    float src[10][3] = {
        {4, 0, 4},
        {4, 2, 0},
        {1, 4, 2},
        {4, 1, 2},
        {1, 0, 2},
        {4, 4, 0},
        {2, 3, 0},
        {2, 4, 0},
        {0, 2, 1},
        {1, 0, 3}
    };
    float kernel[3][3] = {
        {1, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };
    float expected[10][3] = {
        { 6, 14,  2},
        {11, 19, 12},
        {15, 16,  9},
        { 7, 16,  9},
        {13, 18,  7},
        {10, 12,  9},
        {17, 16, 11},
        {11, 10, 10},
        { 9, 11,  9},
        { 2,  7,  3}
    };
    float actual[d1][d2];
    tensor_convolve((float *)src, d1, d2,
                    (float *)kernel, 3, 3,
                    (float *)actual,
                    1);
    for (int i1 = 0; i1 < d1; i1++) {
        for (int i2 = 0; i2 < d2; i2++) {
            float v1 = actual[i1][i2];
            float v2 = expected[i1][i2];
            if (v1 != v2) {
                printf("Mismatch: %.2f != %.2f\n", v1, v2);
                assert(false);
            }
        }
    }
}



int
main(int argc, char *argv[]) {
    PRINT_RUN(test_sub);
    PRINT_RUN(test_cross);
    PRINT_RUN(test_dot);
    PRINT_RUN(test_normalize);
    PRINT_RUN(test_inverse);
    PRINT_RUN(test_mat_mul);
    PRINT_RUN(test_look_at);
    PRINT_RUN(test_to_rad);
    PRINT_RUN(test_to_deg);
    PRINT_RUN(test_perspective);
    PRINT_RUN(test_get_plane);
    PRINT_RUN(test_convolve_1);
    PRINT_RUN(test_convolve_2);
    PRINT_RUN(test_convolve_3);
    return 0;
}
