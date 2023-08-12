// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "benchmark/benchmark.h"
#include "datatypes/common.h"

void
test_identify() {
    benchmark_print_settings();
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_identify);
    return 0;
}
