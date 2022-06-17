// Copyright (C) 2020,2022 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "fastio/fastio.h"

// Run this program like this: ./build/tests/fastio/fastio <<< '1234 '
// All the input needs to be available before this process begins.

int
main(int argc, char *argv[]) {
    fast_io_init();
    int n = fast_io_read_int();
    //fast_io_write_char(n);
    fast_io_write_long(n);
    return 0;
}
