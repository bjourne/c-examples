// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "files/files.h"

static char *exe;

void
test_read() {
    size_t n_bytes = 0;
    assert(files_read(exe, NULL, &n_bytes));
    assert(n_bytes > 0);
    printf("Exe size: %ld\n", n_bytes);

    char *buf = NULL;
    assert(files_read(exe, &buf, NULL));
    assert(buf[n_bytes] == '\0');
    free(buf);
}

int
main(int argc, char *argv[]) {
    exe = argv[0];
    PRINT_RUN(test_read);
}
