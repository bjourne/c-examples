// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#ifndef INT_ARRAY_H
#define INT_ARRAY_H

#include <stdbool.h>
#include <stdio.h>

// Very trivial utility code.
bool int_read(FILE *f, int *value);
int *int1d_read(FILE *f, int n);
int int1d_sum(int *a, int n);
int int1d_max(int *a, int n);

// 2d array utilities.
int int2d_max(int *a, int rows, int cols, int row_stride);

// Pretty prints an int array as a table. If n_points > 0 then the
// given indices in the array will be colorized.
void
int1d_pretty_print_table(int *a, int rows, int cols,
                         int row_stride,
                         int n_points, int points[]);

// Pretty prints a 1-dimensional int-array
void int1d_pretty_print(int *a, int n, int n_cuts, int cuts[]);

#endif
