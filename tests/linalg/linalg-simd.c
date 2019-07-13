#include <assert.h>
#include "datatypes/common.h"
#include "linalg/linalg-simd.h"

void
test_from_vecs() {
    vec3x4 vecs1 = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    assert(vecs1.x[0] == 1);
    assert(vecs1.y[0] == 2);
    assert(vecs1.z[0] == 3);
}

void
test_add() {
    vec3x4 a = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    vec3x4 b = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    vec3x4 c = v3x4_add(a, b);
    assert(c.x[0] == 2);
    assert(c.x[1] == 8);
    assert(c.x[2] == 14);
    assert(c.x[3] == 20);
}

void
test_dot() {
    vec3x4 a = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    vec3x4 b = v3x4_from_vecs((vec3[]){
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
    __m128 dp = v3x4_dot(a, b);
    assert(dp[0] == 14);
    assert(dp[1] == 77);
    assert(dp[2] == 194);
    assert(dp[3] == 365);
}

void
test_f4_abs() {
    __m128 a = { -7.8, 3.2, 0.0, -1.2 };
    float4 b = f4_abs(a);
    assert(approx_eq(b[0], 7.8));
    assert(approx_eq(b[1], 3.2));
    assert(approx_eq(b[2], 0.0));
    assert(approx_eq(b[3], 1.2));
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_from_vecs);
    PRINT_RUN(test_add);
    PRINT_RUN(test_dot);
    PRINT_RUN(test_f4_abs);
    return 0;
}
