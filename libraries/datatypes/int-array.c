// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
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

////////////////////////////////////////////////////////////////////////
// Pretty printing
////////////////////////////////////////////////////////////////////////
#define TERM_WIDTH 167

static int
n_digits(int v) {
    v = v < 0 ? -v : v;
    int digits = 1;
    if (v >= 10)
        digits++;
    if (v >= 100)
        digits++;
    if (v >= 1000)
        digits++;
    if (v >= 10000)
        digits++;
    return digits;
}

// Pretty prints a 2-dimensional int-array
void
int1d_pretty_print_table(int *a, int rows, int cols,
                         int n_points, int points[]) {

    bool ansi = getenv("TERM") != NULL;
    int max = int1d_max(a, rows * cols);
    max = MAX(MAX(max, rows - 1), cols - 1);
    int width = n_digits(max);
    char num_fmt[128], space_fmt[128];
    sprintf(num_fmt, "%%%dd", width);
    sprintf(space_fmt, "%%%ds", width);

    int n_cols_to_print = MIN(TERM_WIDTH / (width + 1) - 1, cols);
    printf(space_fmt, " ");
    printf(" ");
    for (int i = 0; i < n_cols_to_print; i++) {
        printf(num_fmt, i);
        printf(" ");
    }
    if (n_cols_to_print < cols) {
        printf("...");
    }
    printf("\n");
    for (int i = 0; i < rows; i++) {
        printf(num_fmt, i);
        printf(" ");
        for (int j = 0; j < n_cols_to_print; j++) {
            if (ansi) {
                for (int k = 0; k < n_points; k++) {
                    int r = points[2 * k];
                    int c = points[2 * k + 1];
                    if (r == i && c == j) {
                        printf("%c[%dm", '\033', 32);
                    }
                    if (c == i && r == j) {
                        printf("%c[%dm", '\033', 31);
                    }
                }
            }
            printf(num_fmt, a[i*cols + j]);
            if (ansi) {
                printf("%c[0;m", '\033');
            }
            printf(" ");
        }
        if (n_cols_to_print < cols) {
            printf("...");
        }
        printf("\n");
    }
}

// Pretty prints a 1-dimensional int-array
void
int1d_pretty_print(int *a, int n, int n_cuts, int cuts[]) {
    int width = (n <= 100) ? 2 : 3;
    char num_fmt[128];
    sprintf(num_fmt, "%%%dd", width);

    // Guesstimate of the width of the terminal.
    bool ansi = getenv("TERM") != NULL;
    int n_print = MIN(TERM_WIDTH / (width + 1) - 1, n);
    printf("[");
    for (int i = 0; i < n_print; i++) {
        if (ansi) {
            for (int j = 0; j < n_cuts; j++) {
                if (i == cuts[j]) {
                    printf("%c[%dm", '\033', 31 + j);
                }
            }
        }
        printf(num_fmt, a[i]);
        if (ansi) {
            printf("%c[0;m", '\033');
        }
        if (i < n - 1) {
            printf(" ");
        }
    }
    if (n_print < n) {
        printf("...");
    }
    printf("]\n");
}
