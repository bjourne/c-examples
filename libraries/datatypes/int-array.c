// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "int-array.h"

bool
int_read(FILE *f, int *value) {
    int ret = fscanf(f, "%d", value);
    return ret == 1;
}

int *
int1d_read(FILE *f, int n) {
    int *arr = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        if (!int_read(f, &arr[i])) {
            return NULL;
        }
    }
    return arr;
}

int
int1d_sum(int *a, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

int
int1d_max(int *a, int n) {
    int max = INT_MIN;
    for (int i = 0; i < n; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    return max;
}
