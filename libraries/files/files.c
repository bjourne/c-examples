// Copyright (C) 2023 Björn Lindqvist <bjourne@gmail.com>
#include <stdio.h>
#include <stdlib.h>
#include "files.h"

bool
files_read(const char *path, char **data, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    rewind(f);
    if (size) {
        *size = n;
    }
    if (!data) {
        goto ok;
    }
    *data = (char*)malloc(sizeof(char)*(n + 1));
    if (fread(*data, 1, n, f) != n) {
        free(*data);
        return false;
    }
    (*data)[n] = '\0';
 ok:
    fclose(f);
    return true;
}
