// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
__kernel void
matmul(
    const int a_cols, const int b_rows, const int b_cols,
    const __global float* A,
    const __global float* B,
    __global float* C
) {

    const int row = get_global_id(0);
    const int col = get_global_id(1);

    float acc = 0.0f;
    for (int k = 0; k < b_rows; k++) {
        acc += A[a_cols * row + k] * B[k * b_cols + col];
    }

    // Store the result
    C[b_cols * row + col] = acc;
}
