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

    assert(LINALG_APPROX_EQ(m2.d[0][0], 1.0f) &&
           LINALG_APPROX_EQ(m2.d[1][1], 1.0f) &&
           LINALG_APPROX_EQ(m2.d[2][2], 1.0f) &&
           LINALG_APPROX_EQ(m2.d[3][3], 1.0f));

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
    assert(m4_approx_eq(m4, m5));
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

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_sub);
    PRINT_RUN(test_cross);
    PRINT_RUN(test_dot);
    PRINT_RUN(test_normalize);
    PRINT_RUN(test_inverse);
    PRINT_RUN(test_mat_mul);
    return 0;
}
