// Copyright (C) 2022 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "diophantine/diophantine.h"

static void
test_solvable_eqs() {
    int inputs[][3] = {
        {4, 9, -8},
        {-7,  4, -2},
        {3, -4, 8},
        {10, -6, -6}
    };
    for (int i = 0; i < ARRAY_SIZE(inputs); i++) {
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
test_unsolvable_eqs() {
    int inputs[][3] = {
        {2, -2, 1},
        {2, 2, 1}
    };
    for (int i = 0; i < ARRAY_SIZE(inputs); i++) {
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
    for (int i = 0; i < ARRAY_SIZE(inputs); i++) {
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

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_solvable_eqs);
    PRINT_RUN(test_unsolvable_eqs);
    PRINT_RUN(test_constrain_sols);
    return 0;
}
