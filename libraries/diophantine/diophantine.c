// Copyright (C) 2022 Björn Lindqvist <bjourne@gmail.com>

#include "diophantine.h"

#include <math.h>
#include <stdio.h>

static void
gcd_ext(int a, int b, int *g, int *x, int *y) {
    if (b == 0) {
        *g = a;
        *x = 1;
        *y = 0;
    } else {
        int x1, y1;
        gcd_ext(b, a % b, g, &x1, &y1);
        *x = y1;
        *y = x1 - (a / b) * y1;
    }
}


bool
dio_solve_eq(int a, int b, int c, int  *x0, int *xk, int *y0, int *yk) {

    int g, x, y;
    gcd_ext(a, b, &g, &x, &y);
    if (c % g != 0) {
        return false;
    }
    *x0 = x * c / g;
    *y0 = y * c / g;
    *xk = b / g;
    *yk = -a / g;
    return true;
}

// Range of integers, x, satisfying lo <= b + s*x < hi.
void
dio_constrain_to_range(int b, int s, int lo, int hi, int *i_lo, int *i_hi) {
    float f_lo, f_hi;
    if (s < 0) {
        f_lo = (float)(b - hi) / (float)-s;
        f_hi = (float)(b - lo) / (float)-s;

        printf("f_lo, f_hi = %.4f, %.4f\n", f_lo, f_hi);

        if (f_lo == ceil(f_lo)) {
            f_lo += 1;
        }
        if (f_hi == ceil(f_hi)) {
            f_hi +=  1;
        }
    } else {
        f_lo = (float)(lo - b) / (float)s;
        f_hi = (float)(hi - b) / (float)s;
    }
    *i_lo = ceil(f_lo);
    *i_hi = ceil(f_hi);
}
