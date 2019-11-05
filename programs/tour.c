// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "linalg/linalg-io.h"

static void
array2d_pretty_print(int *m, int rows, int cols) {
    printf("    ");
    for (int i = 0; i < cols; i++) {
        printf("%3d ", i);
    }
    printf("\n");
    for (int i = 0; i < rows; i++) {
        printf("%3d ", i);
        for (int j = 0; j < cols; j++) {
            printf("%3d ", m[i*cols + j]);
        }
        printf("\n");
    }
}

typedef struct {
    int *cities;
    int *indices;
} tour;

static tour *
tr_init(int n) {
    tour *tour = malloc(sizeof(tour));
    tour->cities = malloc(sizeof(int) * n);
    tour->indices = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        tour->cities[i] = i;
        tour->indices[i] = i;
    }
    return tour;
}

static void
tr_free(tour *tour) {
    free(tour->cities);
    free(tour->indices);
}

static void
tr_set_city(tour *tour, int idx, int city) {
    tour->cities[idx] = city;
    tour->indices[city] = idx;
}

typedef struct {
    int *distances;
    tour *tour;
    int n;
} tour_optimizer;

static tour_optimizer *
t_opt_init(FILE *stdin) {
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
    array2d_pretty_print(m, n, n);
    opt->distances = m;
    opt->n = n;
    opt->tour = tr_init(n);
    free(cities);
    return opt;
}

static int
t_opt_segment_cost(tour_optimizer *opt, int i, int j) {
    int c1 = opt->tour->cities[i];
    int c2 = opt->tour->cities[j];
    return opt->distances[c1 * opt->n + c2];
}

static int
t_opt_total_cost(tour_optimizer *opt) {
    int tot = 0;
    for (int i = 0; i < opt->n - 1; i++) {
        tot += t_opt_segment_cost(opt, i, i + 1);
    }
    tot += t_opt_segment_cost(opt, opt->n - 1, 0);
    return tot;
}

static void
t_opt_free(tour_optimizer *opt) {
    free(opt->distances);
    tr_free(opt->tour);
    free(opt);
}

static void
t_opt_greedy_tour(tour_optimizer *opt) {
    bool *used = calloc(opt->n, sizeof(bool));
    tr_set_city(opt->tour, 0, 0);

/*     int n = opt->n; */
/*     for (int i = 1; i < n; i++) { */
/*         int best = -1; */
/*         int best_dist = 0xffffff; */
/*         for (int u = 0; u < n; u++) { */
/*             if (!used[u]) { */
/*                 int d = */
/*             } */
/*         } */
/*     } */


    free(used);
}

int
main(int argc, char *argv[]) {
    tour_optimizer *opt = t_opt_init(stdin);
    printf("Total cost %d\n", t_opt_total_cost(opt));
    t_opt_greedy_tour(opt);
    printf("Total cost %d\n", t_opt_total_cost(opt));
    t_opt_free(opt);
}
