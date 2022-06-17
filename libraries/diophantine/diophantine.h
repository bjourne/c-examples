// Copyright (C) 2022 Björn Lindqvist <bjourne@gmail.com>

#ifndef DIOPHANTINE_H
#define DIOPHANTINE_H

#include <stdbool.h>

bool dio_solve_eq(int a, int b, int c, int  *x0, int *xk, int *y0, int *yk);
void dio_constrain_to_range(int b, int s, int lo, int hi,
                            int *i_lo, int *i_hi);

#endif
