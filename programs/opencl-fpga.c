// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// Demonstrates how to run an AOT-compiled kernel on an FPGA.
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "tensors/tensors.h"

#define PE_ROWS                  2
#define PE_COLS                  2

#define DOT_PROD_VECTOR_SIZE     8
#define SCALING_FACTOR 32

#define ROWS_INTERLEAVED         32
#define COLUMNS_INTERLEAVED      32

#define MAT_A_BLOCK_WIDTH           (16 * DOT_PROD_VECTOR_SIZE)
#define MAT_A_BLOCK_HEIGHT          (ROWS_INTERLEAVED   * PE_ROWS)

#define MAT_B_BLOCK_HEIGHT          MAT_A_BLOCK_WIDTH
#define MAT_B_BLOCK_WIDTH           (COLUMNS_INTERLEAVED * PE_COLS)

#define HA (4 * MAT_A_BLOCK_HEIGHT)             // Matrix A height
#define WA (SCALING_FACTOR * MAT_A_BLOCK_WIDTH) // Matrix A width

#define HB WA                                   // Matrix B height
#define WB (4 * MAT_B_BLOCK_WIDTH)              // Matrix B width

#define HC HA                                   // Matrix C height
#define WC WB                                   // Matrix C width

int
main(int argc, char *argv[]) {
    //bool emu = false;
    if (argc == 2 && !strncmp(argv[1], "-emu", strlen("-emu"))) {
        //emu = true;
        printf("using emulator\n");
    }
    tensor *a = tensor_init(2, (int[]){HA, WA});
    tensor *b = tensor_init(2, (int[]){HB, WB});
    tensor *c = tensor_init(2, (int[]){HC, WC});

    printf("A = [%d, %d] B = [%d, %d] C = [%d, %d]\n",
           HA, WA, HB, WB, HC, WC);
    tensor_randrange(a, 10);
    tensor_randrange(b, 10);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}
