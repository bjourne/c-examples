// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "fastio/fastio.h"

int
main(int argc, char *argv[]) {
    fast_io_init();
    int n = fast_io_read_unsigned_int();
    //fast_io_write_char(n);
    fast_io_write_long(n);
    return 0;
}
