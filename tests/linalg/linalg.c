#include <assert.h>
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
    vec3 v2 = {0.81229556, -0.54153037, 0.21661215};
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
            {0.707107, -0.331295, 0.624695, 0},
            {0, 0.883452, 0.468521, 0},
            {-0.707107, -0.331295, 0.624695, 0},
            {-1.63871, -5.747777, -40.400412, 1}
        }
    };
    mat4 m4 = m4_inverse(m3);
    mat4 m5 = {
        {
            { 0.70710695,  0,          -0.70710534,  0},
            {-0.33129495,  0.88345194, -0.33129495,  0},
            { 0.62469512,  0.46852198,  0.62469512,  0},
            {24.49247551, 24.00636673,  22.17499161, 1}
        }
    };
    assert(m4_approx_eq2(m4, m5, 1e-5));
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
    assert(v3_approx_eq(v2, (vec3){0.70833331, 0.80555558, 0.90277779}));
}

void
test_ray_tri_intersect() {
    vec3 orig = {
        24.492475509643554688,
        24.006366729736328125,
        22.174991607666015625
    };
    vec3 dir = {
        -0.582438647747039795,
        -0.430847525596618652,
        -0.689300775527954102
    };
    vec3 v0 = {
        2.079962015151977539,
        8.324080467224121094,
        -4.233458995819091797
    };
    vec3 v1 = {
        1.942253947257995605,
        8.138879776000976562,
        -3.293735027313232422
    };
    vec3 v2 = {
        2.189547061920166016,
        7.210639953613281250,
        -4.343578815460205078
    };
    float t, u, v;
    assert(ray_tri_intersect(orig, dir, v0, v1, v2, &t, &u, &v));
}

void
test_look_at() {
    vec3 eye = {1.0f, 2.0f, 3.0f};
    vec3 center = {4.0f, 5.0f, 6.0f};
    vec3 up = {7.0f, 8.0f, 9.0f};
    mat4 view = m4_look_at(eye, center, up);
    assert(approx_eq2(view.d[0][0], 0.408248276f, 1e-7));
    assert(approx_eq2(view.d[3][3], 1.0, 1e-7));
}

void
test_perspective() {
    mat4 persp = m4_perspective(to_rad(70.0f),
                                500.0f/400.0f,
                                0.1f, 1000.0f);
    assert(approx_eq(persp.d[0][0], 1.142518401));
    assert(persp.d[0][1] == 0.0f);
    assert(persp.d[0][2] == 0.0f);
    assert(persp.d[0][3] == 0.0f);
    assert(persp.d[3][3] == 0.0f);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_sub);
    PRINT_RUN(test_cross);
    PRINT_RUN(test_dot);
    PRINT_RUN(test_normalize);
    PRINT_RUN(test_inverse);
    PRINT_RUN(test_mat_mul);
    PRINT_RUN(test_ray_tri_intersect);
    PRINT_RUN(test_look_at);
    PRINT_RUN(test_perspective);
    return 0;
}
