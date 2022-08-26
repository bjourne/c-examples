// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
//
// | Target |   Size |  Time | Compiler |Opt     | Comment
// |  Xeon  |   1024 |  5.76 |    clang | -O3    |
// |  Xeon  |   1024 |  5.72 |    clang | -O2    |

// |  Xeon  |   1024 |  2.87 |    clang | -O3    | i,j,k
// |  Xeon  |   1024 |  0.20 |    clang | -O2    | i,k,j
// |  Xeon  |   1024 |  2.83 |    clang | -O2    | i,j,k
// |  Xeon  |   1024 |  5.86 |    gcc   | -O3    |

// |  Xeon  |   2048 |   1.51 |    clang | -O2    | ikj
// |  Xeon  |   2048 |   0.64 |    clang | -O2    | ikj, omp(3)
// |  Xeon  |   2048 |  28.41 |    clang | -O2    | ijk
// |  Xeon  |   2048 |   1.81 |    clang | -O2    | kij
// |  Xeon  |   2048 |  53.07 |    clang | -O2    | kji
// |  Xeon  |   2048 |  27.72 |    clang | -O2    | jik
// |  Xeon  |   2048 |  67.13 |    clang | -O2    | jki

// |  Xeon  |   4096 |   4.61 |    clang | -O2    | ikj, omp static(1)
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "linalg/linalg.h"
#include "tensors/tensors.h"

#define SIZE 4096

void
test_mul_perf() {
    int a_rows = SIZE;
    int a_cols = SIZE;
    int b_rows = SIZE;
    int b_cols = SIZE;

    tensor *a = tensor_init(2, (int[]){a_rows, a_cols});
    tensor *b = tensor_init(2, (int[]){b_rows, b_cols});
    tensor *c = tensor_init(2, (int[]){a_rows, b_cols});

    tensor_randrange(a, 100);
    tensor_randrange(b, 100);

    tensor_multiply(a, b, c);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_mul_perf);
}
