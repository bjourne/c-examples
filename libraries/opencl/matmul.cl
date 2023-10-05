// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include "libraries/opencl/matmul.h"

// See https://cnugteren.github.io/tutorial/pages/page5.html
__kernel void
matmul_tiled_simd(
    const int M,
    const int N,
    const int K,
    const __global float* A,
    const __global float* B,
    __global float* C
) {
    // Thread identifiers
    const int lr = get_local_id(0);
    const int lc = get_local_id(1);
    const int gr = TILE_SIZE * get_group_id(0) + lr;
    const int gc = TILE_SIZE * get_group_id(1) + lc;

    printf("%d %d\n", gr, gc);

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float At[TILE_SIZE][TILE_SIZE];
    __local float Bt[TILE_SIZE][TILE_SIZE];

    // Initialise the accumulation registers
    float acc[WORK_SIZE];
    for (uint w = 0; w < WORK_SIZE; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    const uint n_tiles = K / TILE_SIZE;
    for (uint t = 0; t < n_tiles; t++) {

        // Load one tile of A and B into local memory
        const uint tr = TILE_SIZE * t + lr;
        const uint tc = TILE_SIZE * t + lc;
        for (int w=0; w < WORK_SIZE; w++) {
            At[WORK_SIZE * lr + w][lc] = A[(gr * WORK_SIZE + w) * K + tc];
            Bt[WORK_SIZE * lr + w][lc] = B[(tr * WORK_SIZE + w) * N + gc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            for (int w = 0; w < WORK_SIZE; w++) {
                float a = At[WORK_SIZE * lr + w][k];
                float b = Bt[k][lc];
                acc[w] += a * b;
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint w=0; w < WORK_SIZE; w++) {
        C[(WORK_SIZE * gr + w) * N + gc] = acc[w];
    }
}


// See https://cnugteren.github.io/tutorial/pages/page4.html
__kernel void
matmul_tiled(
    const int M,
    const int N,
    const int K,
    const __global float* A,
    const __global float* B,
    __global float* C
) {
    // Local row and col
    const uint lr = get_local_id(0);
    const uint lc = get_local_id(1);

    // Global row and col
    const uint gr = TILE_SIZE * get_group_id(0) + lr;
    const uint gc = TILE_SIZE * get_group_id(1) + lc;

    __local float At[TILE_SIZE][TILE_SIZE];
    __local float Bt[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    const uint n_tiles = K / TILE_SIZE;

    for (uint t = 0; t < n_tiles; t++) {
        // Load one tile of A and B
        const uint tr = TILE_SIZE*t + lr;
        const uint tc = TILE_SIZE*t + lc;

        At[lr][lc] = A[gr * K + tc];
        Bt[lr][lc] = B[tr * N + gc];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += At[lr][k] * Bt[k][lc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[gr*N + gc] = acc;
}

__kernel void
matmul_naive(
    const int M, const int N, const int K,
    const __global float* A,
    const __global float* B,
    __global float* C
) {

    // Row and col of C
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += A[K * row + k] * B[k * N + col];
    }

    // Store the result
    C[N * row + col] = acc;
}
