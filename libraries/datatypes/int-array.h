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

#endif
