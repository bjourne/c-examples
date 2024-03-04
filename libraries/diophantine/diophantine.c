// Copyright (C) 2022, 2024 Björn A. Lindqvist <bjourne@gmail.com>

#include "datatypes/common.h"
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


// Find solution to ax + by = c
bool
dio_solve_eq(int a, int b, int c,
             int *x0, int *xk, int *y0, int *yk) {
    if (a == 0 && b == 0) {
        *x0 = 0;
        *y0 = 0;
        *xk = 1;
        *yk = 1;
        return c == 0;
    }
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

// Check if the Diophantine equation ax + by = c has a solution where
// both x and y are in coeffs. The coefficients must be stored with +
// 2.
bool
dio_solve_eq_with(int a, int b, int c,
                  hashset *coeffs, int *res_x, int *res_y) {
    int x0, xk, y0, yk;
    if (!dio_solve_eq(a, b, c, &x0, &xk, &y0, &yk)) {
        return false;
    }
    HS_FOR_EACH_ITEM(coeffs, {
            int x = p - 2;
            if ((x - x0) % xk == 0) {
                int n = (x - x0) / xk;
                int y = y0 + yk * n;
                if (hs_in_p(coeffs, y + 2)) {
                    *res_x = x;
                    *res_y = y;
                    return true;
                }
            }
        });
    return false;
}

// Range of integers, x, satisfying lo <= b + s*x < hi.
void
dio_constrain_to_range(int b, int s, int lo, int hi, int *i_lo, int *i_hi) {
    float f_lo, f_hi;
    if (s < 0) {
        f_lo = (float)(b - hi) / (float)-s;
        f_hi = (float)(b - lo) / (float)-s;

        //printf("f_lo, f_hi = %.4f, %.4f\n", f_lo, f_hi);

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

// All two-sets {a, b}, ordered by their sum, where 0 <= a, b, a + b
// < max_sum and a, b are integers.
int *dio_2sets_ordered_by_sum(int max_sum, size_t *n) {
    for (int sum = 0; sum < max_sum; sum++) {
        *n += sum / 2 + 1;
    }
    int *arr = malloc(2 * sizeof(int) * *n);
    int c = 0;

    for (int sum = 0; sum < max_sum; sum++) {
        //printf("== %d\n", sum);
        int lo = MAX(sum - max_sum, 0);
        int hi = MIN(max_sum + 1, sum / 2 + 1);
        for (int a = lo; a < hi; a++) {
            arr[2 * c + 0] = a;
            arr[2 * c + 1] = sum - a;
            //printf("%2d %2d\n", a, sum - a);
            c++;
        }
    }
    return arr;


    //#assert(cnt == max_sum
    //printf("cnt = %d, %d\n", cnt, *n);
}
