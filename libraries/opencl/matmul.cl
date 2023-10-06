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
    // row in [0..TILE_SIZE)
    const int lr = get_local_id(0);
    // col in [0..TILE_SIZE/WPT)
    const int lc = get_local_id(1);

    // Row ID of C (0..M)
    const int gr = TILE_SIZE*get_group_id(0) + lr;
    // Col ID of C (0..N)
    const int gc = TILE_SIZE*get_group_id(1) + lc;

    __local float At[TILE_SIZE][TILE_SIZE];
    __local float Bt[TILE_SIZE][TILE_SIZE];

    // Initialise the accumulation registers
    float acc[WPT];
    for (uint w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    const uint n_tiles = K / TILE_SIZE;
    for (uint t=0; t < n_tiles; t++) {

        // Load one tile of A and B into local memory
        const uint tbase = TILE_SIZE * t;
        for (int w=0; w < WPT; w++) {
            const uint tr = tbase + lr;
            const uint tc = tbase + lc;

            const uint addr_a = K * gr + tc + w * RTS;
            const uint addr_b = N * tr + gc + w * RTS;

            At[lc + w * RTS][lr] = A[addr_a];
            Bt[lc + w * RTS][lr] = B[addr_b];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (uint k=0; k < TILE_SIZE; k++) {
            for (uint w=0; w < WPT; w++) {
                acc[w] += At[k][lr] * Bt[lc + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (uint w = 0; w < WPT; w++) {
        C[gr * N + gc + w*RTS] = acc[w];
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
