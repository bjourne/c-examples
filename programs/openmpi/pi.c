// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Approximate pi using the dartboard algorithm. Run with:
//
//   mpirun -n 8 ./build/programs/openmpi/pi
//
// From https://hpc-tutorials.llnl.gov/mpi/
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "datatypes/common.h"
#include "random/random.h"

#define N_DARTS (1000)
#define N_ROUNDS (1000)

static double
estimate(size_t n_darts) {
    size_t n_hits = 0;
    for (uint32_t i = 0; i < n_darts; i++) {
        double x = rnd_pcg32_rand_double_0_to_1();
        double y = rnd_pcg32_rand_double_0_to_1();
        if (x * x + y * y <= 1.0) {
            n_hits++;
        }
    }
    return (double)(4 * n_hits) / n_darts;
}

int
main(int argc, char *argv[]) {
    rnd_pcg32_seed(nano_count(), nano_count());

    int32_t n_tasks, task_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    double est = 0.0;
    for (uint32_t i = 0; i < N_ROUNDS; i++) {
        double sum_est = 0.0;
        double my_est = estimate(N_DARTS);

        int rc = MPI_Reduce(&my_est, &sum_est, 1, MPI_DOUBLE, MPI_SUM,
                            0, MPI_COMM_WORLD);
        assert(rc == MPI_SUCCESS);

        if (task_id == 0) {
            // Complicated, but appears to work
            est = ((est * i) + (sum_est / n_tasks)) / (i + 1);
            double rel_err = 100 * fabs(M_PI - est) / M_PI;
            printf("Estimate : %12.8f (err: %.5f%%)\n", est, rel_err);
        }
    }
    MPI_Finalize();
    return 0;
}
