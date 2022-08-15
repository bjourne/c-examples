// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"

char *fname = NULL;

void
test_from_png() {
    tensor *t = tensor_read_png(fname);

    assert(t);
    printf("%d\n", t->error_code);
    assert(t->error_code == TENSOR_ERR_NONE);

    assert(tensor_write_png(t, "out_01.png"));
    assert(t->error_code == TENSOR_ERR_NONE);

    tensor_free(t);
}

void
test_pick_channel() {
    tensor *t1 = tensor_read_png(fname);
    assert(t1);
    assert(t1->error_code == TENSOR_ERR_NONE);

    int height = t1->dims[1];
    int width = t1->dims[2];

    tensor *t2 = tensor_allocate(3, 3, height, width);
    tensor_fill(t2, 0.0);
    for (int c = 0; c < 1; c++) {
        float *dst = &t2->data[c * height * width];
        memcpy(dst, t1->data, height * width * sizeof(float));
    }
    assert(tensor_write_png(t2, "out_02.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);

    tensor_free(t1);
    tensor_free(t2);
}

void
test_conv2d() {
    tensor *t1 = tensor_read_png(fname);
    assert(t1);
    assert(t1->error_code == TENSOR_ERR_NONE);

    int height = t1->dims[1];
    int width = t1->dims[2];
    tensor *t2 = tensor_allocate(3, 3, height, width);

    float w = 1/8.0;

    // Blur red and green channels and filter blue channel.
    float kernel_data[3][3][3][3] = {
        {
            {
                {w, w, w},
                {w, 0, w},
                {w, w, w}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            }
        },
        {
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {w, w, w},
                {w, 0, w},
                {w, w, w}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            }
        },
        {
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            },
            {
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
            }
        }
    };
    tensor *kernel = tensor_allocate(4, 3, 3, 3, 3);
    memcpy(kernel->data, kernel_data,
           tensor_n_elements(kernel) *  sizeof(float));


    tensor_conv2d(t1, kernel, t2, 1, 1);

    assert(tensor_write_png(t2, "out_04.png"));
    assert(t2->error_code == TENSOR_ERR_NONE);
    tensor_free(t2);
    tensor_free(t1);
    tensor_free(kernel);
}

int
main(int argc, char *argv[]) {
    fname = argv[1];
    PRINT_RUN(test_from_png);
    PRINT_RUN(test_pick_channel);
    PRINT_RUN(test_conv2d);
}
