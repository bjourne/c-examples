// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "datatypes/hashset.h"
#include "diophantine/diophantine.h"
#include "random/random.h"

static void
test_solvable_eqs() {
    int inputs[][3] = {
        {4, 9, -8},
        {-7,  4, -2},
        {3, -4, 8},
        {10, -6, -6},
        {4, 18, 10}
    };
    for (size_t i = 0; i < ARRAY_SIZE(inputs); i++) {
        int x0, xk, y0, yk;
        int a = inputs[i][0];
        int b = inputs[i][1];
        int c = inputs[i][2];
        dio_solve_eq(a, b, c, &x0, &xk, &y0, &yk);
        for (int j = -10; j < 10; j++) {
            assert(a*(x0 + j*xk) + b*(y0 + j*yk) == c);
        }
    }
}

static void
test_zero_ab() {
    int x0, xk, y0, yk;

    // 0x + 0y = 0
    assert(dio_solve_eq(0, 0, 0, &x0, &xk, &y0, &yk));

    // 30x + 0y = 0
    assert(dio_solve_eq(30, 0, 0, &x0, &xk, &y0, &yk));
    assert(x0 == 0 && xk == 0);

    assert(dio_solve_eq(0, 30, 0, &x0, &xk, &y0, &yk));
    assert(y0 == 0 && yk == 0);


    assert(!dio_solve_eq(0, 0, 30, &x0, &xk, &y0, &yk));
}

static void
test_unsolvable_eqs() {
    int inputs[][3] = {
        {2, -2, 1},
        {2, 2, 1}
    };
    for (size_t i = 0; i < ARRAY_SIZE(inputs); i++) {
        int a = inputs[i][0];
        int b = inputs[i][1];
        int c = inputs[i][2];
        assert(!dio_solve_eq(a, b, c, NULL, NULL, NULL, NULL));
    }
}

static void
test_constrain_sols() {
    int inputs[][4] = {
        {12, -1, 0, 10},
        {12, -1, 5, 10},
        {-6, -5, 0, 42},
        {-7, -5, 0, 3},
        {0, 1, 0, 16},
        {-2, 4, 0, 20},
        {-4, 7, 0, 8},
        {3, 3, 0, 28}
    };
    int ranges[][2] = {
        {3, 13},
        {3, 8},
        {-9, -1},
        {-1, -1},
        {0, 16},
        {1, 6},
        {1, 2},
        {-1, 9}
    };
    for (size_t i = 0; i < ARRAY_SIZE(inputs); i++) {
        int r_lo, r_hi;
        int b = inputs[i][0];
        int s = inputs[i][1];
        int lo = inputs[i][2];
        int hi = inputs[i][3];
        dio_constrain_to_range(b, s, lo, hi, &r_lo, &r_hi);
        assert(r_lo == ranges[i][0]);
        assert(r_hi == ranges[i][1]);
    }
}

void
test_2sets() {
    size_t n = 0;
    int (*arr)[2] = (int (*)[2])dio_2sets_ordered_by_sum(5, &n);
    assert(n == 9);
    assert(arr[0][0] == 0);
    assert(arr[0][1] == 0);

    assert(arr[8][0] == 2);
    assert(arr[8][1] == 2);
    free(arr);
}

void
bruteforce(int Z_MAX, size_t nA, int A[], size_t nC, int C[], int R[]) {
    for (size_t i = 0; i < nC; i++) {
        R[i] = 1 << 20;
        int c = C[i];
        for (size_t j = 0; j < nA; j++) {
            for (size_t k = 0; k < nA; k++) {
                for (int a = 0; a <= Z_MAX; a++) {
                    for (int b = 0; b <= Z_MAX; b++) {
                        if (a * A[j] + b * A[k] == c) {
                            R[i] = MIN(R[i], a + b);
                        }
                    }
                }
            }
        }
        if (R[i] > Z_MAX) {
            R[i] = -1;
        }
    }
}

void
betterforce(int Z_MAX, size_t nA, int A[], size_t nC, int C[], int R[]) {
    hashset *coeffs = hs_init();
    for (size_t i = 0; i < nA; i++) {
        hs_add(coeffs, A[i] + 2);
    }
    size_t n_pairs = 0;
    int (*pairs)[2] = (int (*)[2])dio_2sets_ordered_by_sum(
        Z_MAX + 1, &n_pairs);
    for (size_t i = 0; i < nC; i++) {
        int c = C[i];
        R[i] = -1;
        for (size_t j = 0; j < n_pairs; j++) {
            int a = pairs[j][0];
            int b = pairs[j][1];
            int x, y;
            if (dio_solve_eq_with(a, b, c, coeffs, &x, &y)) {
                R[i] = a + b;
                printf("%3d*%5d + %3d*%5d = %5d\n", a, x, b, y, c);
                break;
            }
        }
    }
    free(pairs);
    hs_free(coeffs);
}

void
test_constrained_eqs() {
    int Z_MAX = 20;
    size_t nC = 500;
    int *C = malloc(nC * sizeof(int));
    rnd_pcg32_rand_range_fill((uint32_t *)C, 10000, nC);

    size_t nA = 50;
    int *A = malloc(nA * sizeof(int));
    rnd_pcg32_rand_range_fill((uint32_t *)A, 20000, nA);

    int *R1 = malloc(nC * sizeof(int));
    int *R2 = malloc(nC * sizeof(int));

    PRINT_CODE_TIME({
            bruteforce(Z_MAX, nA, A, nC, C, R1);
        }, "Bruteforce  took %8.2fs\n");

    PRINT_CODE_TIME({
            betterforce(Z_MAX, nA, A, nC, C, R2);
        }, "Betterforce took %8.2fs\n");
    for (size_t i = 0; i < nC; i++) {
        assert(R1[i] == R2[i]);
    }
    free(C);
    free(R1);
    free(R2);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_solvable_eqs);
    PRINT_RUN(test_zero_ab);
    PRINT_RUN(test_unsolvable_eqs);
    PRINT_RUN(test_constrain_sols);
    PRINT_RUN(test_2sets);
    PRINT_RUN(test_constrained_eqs);
    return 0;
}
