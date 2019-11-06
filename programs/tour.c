// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "datatypes/array.h"
#include "datatypes/int-array.h"
#include "linalg/linalg-io.h"

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

typedef struct {
    int *cities;
    int *indices;
} tour;

tour *
tr_init(int n) {
    tour *tr = malloc(sizeof(tour));
    tr->cities = malloc(sizeof(int) * n);
    tr->indices = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        tr->cities[i] = i;
        tr->indices[i] = i;
    }
    return tr;
}

void
tr_free(tour *tr) {
    free(tr->cities);
    free(tr->indices);
    free(tr);
}

static void
tr_set_city(tour *tour, int idx, int city) {
    tour->cities[idx] = city;
    tour->indices[city] = idx;
}

typedef struct {
    int *costs;
    int *neighbors;
    tour *tour;
    int n;
} tour_optimizer;

static inline int
t_opt_cost(tour_optimizer *opt, int c1, int c2) {
    return opt->costs[c1 * opt->n + c2];
}

static inline int
t_opt_cost_segment(tour_optimizer *opt, int i, int j) {
    int c1 = opt->tour->cities[i];
    int c2 = opt->tour->cities[j];
    return t_opt_cost(opt, c1, c2);
}

static int
t_opt_cost_total(tour_optimizer *opt) {
    int tot = 0;
    for (int i = 0; i < opt->n - 1; i++) {
        tot += t_opt_cost_segment(opt, i, i + 1);
    }
    tot += t_opt_cost_segment(opt, opt->n - 1, 0);
    return tot;
}

typedef struct {
    tour_optimizer *opt;
    int city;
} neighbor_key_context;

static int
neighbor_key_fun(void *ctx, const void *a) {
    neighbor_key_context *obj = (neighbor_key_context *)ctx;
    return t_opt_cost(obj->opt, obj->city, *(int *)a);
}

tour_optimizer*
t_opt_init(FILE *inf) {
    tour_optimizer *opt = malloc(sizeof(tour_optimizer));

    int n;
    vec2 *cities = v2_array_read(stdin, &n);

    // Create cost matrix
    int *m = malloc(sizeof(int) * n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float d = v2_distance(cities[i], cities[j]);
            m[i * n + j] = (int)(d + 0.5);
        }
    }
    opt->costs = m;
    opt->n = n;
    opt->tour = tr_init(n);
    free(cities);

    // Create neighbor lists
    m = malloc(sizeof(int) * n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = j;
        }
        neighbor_key_context ctx = { opt, i };
        array_qsort_with_key(&m[i * n], n, sizeof(int),
                             neighbor_key_fun, (void *)&ctx);
    }
    opt->neighbors = m;
    return opt;
}

void
t_opt_free(tour_optimizer *opt) {
    free(opt->costs);
    free(opt->neighbors);
    tr_free(opt->tour);
    free(opt);
}

void
t_opt_greedy_tour(tour_optimizer *opt) {
    bool *used = calloc(opt->n, sizeof(bool));
    tour *tour = opt->tour;
    int n = opt->n;
    tr_set_city(tour, 0, 0);
    used[0] = true;
    for (int i = 1; i < n; i++) {
        int best = -1;
        int best_cost = 0xffffff;
        for (int u = 0; u < n; u++) {
            if (!used[u]) {
                int c = t_opt_cost(opt, tour->cities[i - 1], u);
                if (c < best_cost) {
                    best = u;
                    best_cost = c;
                }
            }
        }
        tr_set_city(opt->tour, i, best);
        used[best] = true;
    }
    free(used);
}

void
t_opt_pretty_print_tour(tour_optimizer *opt, int n_cuts, int cuts[]) {
    printf("%6d ", t_opt_cost_total(opt));
    int1d_pretty_print(opt->tour->cities, opt->n, n_cuts, cuts);
}

void
t_opt_pretty_print_costs(tour_optimizer *opt) {
    // Points of interests
    int n = opt->n;
    int *points = malloc(sizeof(int) * n * 2);
    for (int i = 0; i < n - 1; i++) {
        points[2 * i] = opt->tour->cities[i];
        points[2 * i + 1] = opt->tour->cities[i + 1];
    }
    points[2 * (n - 1)] = opt->tour->cities[n - 1];
    points[2 * (n - 1) + 1] = opt->tour->cities[0];
    int1d_pretty_print_table(opt->costs, n, n, n, points);
    free(points);
    int1d_pretty_print_table(opt->neighbors, n, n, 0, NULL);
}

int
main(int argc, char *argv[]) {
    tour_optimizer *opt = t_opt_init(stdin);
    t_opt_pretty_print_tour(opt, 0, NULL);
    t_opt_greedy_tour(opt);
    t_opt_pretty_print_tour(opt, 0, NULL);
    t_opt_pretty_print_costs(opt);
    t_opt_free(opt);
}
