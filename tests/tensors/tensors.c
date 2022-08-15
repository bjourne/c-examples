// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "tensors/tensors.h"

char *fname = NULL;

void
test_from_png() {
    tensor *t = tensor_read_png(fname);

    assert(t);
    assert(t->error_code == TENSOR_ERR_NONE);

    assert(tensor_write_png(t, "foo.png"));
    assert(t->error_code == TENSOR_ERR_NONE);

    tensor_free(t);
}

int
main(int argc, char *argv[]) {
    fname = argv[1];
    PRINT_RUN(test_from_png);
}
