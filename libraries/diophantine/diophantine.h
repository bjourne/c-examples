// Copyright (C) 2022, 2024 Björn Lindqvist <bjourne@gmail.com>

#ifndef DIOPHANTINE_H
#define DIOPHANTINE_H

#include <stdbool.h>
#include "datatypes/hashset.h"

bool dio_solve_eq(int a, int b, int c, int  *x0, int *xk, int *y0, int *yk);
bool dio_solve_eq_with(int a, int b, int c,
                       hashset *coeffs, int *res_x, int *res_y);
void dio_constrain_to_range(int b, int s, int lo, int hi,
                            int *i_lo, int *i_hi);
int *dio_2sets_ordered_by_sum(int sum, size_t *n);

#endif
