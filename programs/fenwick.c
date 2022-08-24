// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "fastio/fastio.h"

int
fenwick_solve(int n, int q) {
    long *arr = (long *)calloc(n + 1, sizeof(long));
    for (int k = 0; k < q; k++) {
        char c = fast_io_read_char();
        if (c == '+') {
            fast_io_read_char();
            int i = (int)fast_io_read_unsigned_int() + 1;
            fast_io_read_char();
            int inc = fast_io_read_int();
            fast_io_read_char();
            while (i <= n) {
                arr[i] += inc;
                i += i & (-i);
            }
        } else {
            fast_io_read_char();
            int i = (int)fast_io_read_unsigned_int();
            fast_io_read_char();
            long sum = 0;
            while (i > 0) {
                sum += arr[i];
                i -= i & (-i);
            }
            fast_io_write_long(sum);
            fast_io_write_char('\n');
        }
    }
    free(arr);
    return 0;
}

int
main(int argc, char *argv[]) {
    fast_io_init();
    unsigned int n = fast_io_read_unsigned_int();
    fast_io_read_char();
    unsigned int q = fast_io_read_unsigned_int();
    fast_io_read_char();
    return fenwick_solve(n, q);
}
