// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Solve a simplified two-dimensional heat equation using OpenMPI. Run
// with:
//
//   mpirun --oversubscribe -n 5 ./build/programs/openmpi/heat
//
// From https://hpc-tutorials.llnl.gov/mpi/
#include <assert.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"

// Size of grid
#define GRID_X  15
#define GRID_Y  15

// Dissipation parameters
#define C_X     0.1
#define C_Y     0.1

// Simulation steps
#define N_STEPS 10

// Pretty printing
#define GRID_FMT "%6.2f"

// Message tags
#define MSG_BEGIN       1
#define MSG_ROW_UP      2
#define MSG_ROW_DOWN    3
#define MSG_DONE        4

static void
send_rows(tensor *t, int ofs, int n_rows, int dst, int tag) {
    MPI_Send(&t->data[ofs * GRID_X], n_rows * GRID_X, MPI_FLOAT,
             dst, tag, MPI_COMM_WORLD);
}

static void
recv_rows(tensor *t, int ofs, int n_rows, int src, int tag) {
    MPI_Status st;
    MPI_Recv(&t->data[ofs * GRID_X], n_rows * GRID_X, MPI_FLOAT,
             src, tag, MPI_COMM_WORLD, &st);
}

static void
update_rows(tensor *t0, tensor *t1, int ofs0, int ofs1) {
    float *d0 = t0->data;
    float *d1 = t1->data;
    for (int y = ofs0; y < ofs1; y++) {
        for (int x = 1; x < GRID_X - 1; x++) {
            size_t addr = y * GRID_X + x;
            float at = d0[addr];
            float north = d0[addr - GRID_X];
            float south = d0[addr + GRID_X];
            float west = d0[addr - 1];
            float east = d0[addr + 1];

            d1[addr] = at
                + C_X * (east + west - 2 * at)
                + C_Y * (south + north - 2 *at);
        }
    }
}

int
main(int argc, char *argv[]) {
    uint32_t n_tasks, my_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, (int32_t *)&n_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, (int32_t *)&my_id);
    uint32_t n_workers = n_tasks - 1;

    tensor *U0 = tensor_init(2, (int[]){GRID_Y, GRID_X});
    tensor *U1 = tensor_init(2, (int[]){GRID_Y, GRID_X});
    tensor_fill_const(U0, 0);
    tensor_fill_const(U1, 0);

    if (my_id == 0) {
        // Initialize grid
        U0->data[(GRID_Y / 2) * GRID_X + GRID_X / 2] = 500;
        tensor_print(U0, GRID_FMT, false);

        // Distribute work to workers
        uint32_t div = GRID_Y / n_workers;
        uint32_t rem = GRID_Y % n_workers;
        uint32_t ofs0 = 0;
        uint32_t offsets[n_workers][2];
        for (uint32_t i = 0; i < n_workers; i++) {
            uint32_t n_rows = i <= rem ? div + 1 : div;
            uint32_t ofs1 = MIN(ofs0 + n_rows, GRID_Y);
            uint32_t dst = i + 1;

            offsets[i][0] = ofs0;
            offsets[i][1] = ofs1;

            // Send config to workers
            MPI_Send(&ofs0, 1, MPI_INT, dst, MSG_BEGIN, MPI_COMM_WORLD);
            MPI_Send(&ofs1, 1, MPI_INT, dst, MSG_BEGIN, MPI_COMM_WORLD);

            // Send data
            send_rows(U0, ofs0, ofs1 - ofs0, dst, MSG_BEGIN);
            ofs0 = ofs1;
        }
        for (uint32_t i = 0; i < n_workers; i++) {
            uint32_t src = i + 1;
            uint32_t ofs0 = offsets[i][0];
            uint32_t ofs1 = offsets[i][1];
            recv_rows(U0, ofs0, ofs1 - ofs0, src, MSG_DONE);
        }
        tensor_print(U0, GRID_FMT, false);
    }

    if (my_id != 0) {
        uint32_t ofs0, ofs1;
        MPI_Status st;
        MPI_Recv(&ofs0, 1, MPI_INT, 0, MSG_BEGIN, MPI_COMM_WORLD, &st);
        MPI_Recv(&ofs1, 1, MPI_INT, 0, MSG_BEGIN, MPI_COMM_WORLD, &st);

        uint32_t start = ofs0 == 0 ? 1 : ofs0;
        uint32_t end = ofs1 == GRID_Y ? GRID_Y - 1 : ofs1;

        int up_id = my_id - 1;
        int down_id = my_id < n_workers ? my_id + 1 : 0;

        // Receive my share of the grid
        uint32_t n_rows = ofs1 - ofs0;
        recv_rows(U0, ofs0, n_rows, 0, MSG_BEGIN);

        // Begin simulating
        for (int i = 0; i < N_STEPS; i++) {
            if (up_id) {
                // Send edge row to upper neighbor.
                send_rows(U0, ofs0, 1, up_id, MSG_ROW_UP);
                recv_rows(U0, ofs0 - 1, 1, up_id, MSG_ROW_DOWN);
            }
            if (down_id) {
                // Send edge to down neighbor.
                send_rows(U0, ofs1 - 1, 1, down_id, MSG_ROW_DOWN);
                recv_rows(U0, ofs1, 1, down_id, MSG_ROW_UP);
            }
            update_rows(U0, U1, start, end);
            tensor *tmp = U0;
            U0 = U1;
            U1 = tmp;
        }
        send_rows(U0, ofs0, n_rows, 0, MSG_DONE);
    }

    tensor_free(U0);
    tensor_free(U1);
    MPI_Finalize();
    return 0;
}
