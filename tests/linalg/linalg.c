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

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_sub);
    PRINT_RUN(test_cross);
    PRINT_RUN(test_dot);
    PRINT_RUN(test_normalize);
    return 0;
}
