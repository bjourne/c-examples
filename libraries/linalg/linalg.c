#include <stdio.h>
#include "linalg.h"

extern inline vec3 v3_add(vec3 l, vec3 r);
extern inline vec3 v3_sub(vec3 l, vec3 r);
extern inline vec3 v3_cross(vec3 l, vec3 r);
extern inline float v3_dot(vec3 l, vec3 r);
extern inline vec3 v3_normalize(vec3 in);

extern inline bool v3_approx_eq(vec3 l, vec3 r);

extern inline float to_rad(const float deg);
extern inline float to_deg(const float rad);

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

static void
print_float(float f, int n_dec) {
    char buf[256];
    sprintf(buf, "%%.%df", n_dec);
    printf(buf, f);
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

bool
m4_approx_eq(mat4 l, mat4 r) {
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            if (!LINALG_APPROX_EQ(l.d[y][x], r.d[y][x])) {
                printf("diff %d %d\n", y, x);
                return false;
            }
        }
    }
    return true;
}

extern inline vec3 m4_mul_v3(mat4 m, vec3 v);
