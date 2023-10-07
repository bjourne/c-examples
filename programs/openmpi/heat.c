// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Solve a simplified two-dimensional heat equation using OpenMPI.
//
// From https://hpc-tutorials.llnl.gov/mpi/
#include <assert.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"

#define GRID_X  10
#define GRID_Y  10

// Message tags
#define MSG_BEGIN       1

int
main(int argc, char *argv[]) {
    uint32_t n_tasks, task_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, (int32_t *)&n_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, (int32_t *)&task_id);
    uint32_t n_workers = n_tasks - 1;

    tensor *U = tensor_init(2, (int[]){GRID_Y, GRID_X});

    if (task_id == 0) {
        assert(n_workers == 4);

        // Initialize grid
        tensor_fill_rand_range(U, 10);
        tensor_print(U, "%5.2f", false);

        // Distribute work to workers
        uint32_t div = GRID_Y / n_workers;
        uint32_t rem = GRID_Y % n_workers;
        uint32_t ofs0 = 0;
        for (uint32_t i = 0; i < n_workers; i++) {
            uint32_t n_rows = i <= rem ? div + 1 : div;
            uint32_t ofs1 = MIN(ofs0 + n_rows, GRID_Y);

            // Send config to workers
            MPI_Send(&ofs0, 1, MPI_INT, i + 1, MSG_BEGIN, MPI_COMM_WORLD);
            MPI_Send(&ofs1, 1, MPI_INT, i + 1, MSG_BEGIN, MPI_COMM_WORLD);

            ofs0 = ofs1;
        }
    }

    if (task_id != 0) {
        uint32_t ofs0, ofs1;
        MPI_Status st;
        MPI_Recv(&ofs0, 1, MPI_INT, 0, MSG_BEGIN, MPI_COMM_WORLD, &st);
        MPI_Recv(&ofs1, 1, MPI_INT, 0, MSG_BEGIN, MPI_COMM_WORLD, &st);
        printf("%u: Got job [%u, %u)\n", task_id, ofs0, ofs1);
    }

    tensor_free(U);
    MPI_Finalize();
    return 0;
}
