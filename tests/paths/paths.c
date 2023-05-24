// Copyright (C) 2019, 2023 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <string.h>
#include "datatypes/common.h"
#include "paths/paths.h"

void
test_basename() {
    char* tests[][2] = {
        {"C:\\what\\is\\it", "it"},
        {"only.txt", "only.txt"},
        {"C:\\yes\\", ""},
        {"/C/foo.txt", "foo.txt"},
        {"", ""}
    };
    for (int i = 0; i < ARRAY_SIZE(tests); i++) {
        const char *res = paths_basename(tests[i][0]);
        assert(!strcmp(tests[i][1], res));
    }
}

void
test_dirname() {
    char* tests[][2] = {
        {"what is this", ""},
        {"", ""},
        {"///", "/"},
        {"////////", "/"},
        {"/omy", "/"},
        {"/here/", "/here"},
        {"/this/is/simple", "/this/is"},
        {"/this/is/////simple", "/this/is"},
        {"this/here", "this"}
    };
    for (int i = 0; i < ARRAY_SIZE(tests); i++) {
        char *res = paths_dirname(tests[i][0]);
        assert(!strcmp(tests[i][1], res));
        free(res);
    }
}

void
test_stem() {
    char* tests[][2] = {
        {"libraries/opencl/matmul.cl", "matmul"}
    };
    for (int i = 0; i < ARRAY_SIZE(tests); i++) {
        char *res = paths_stem(tests[i][0]);
        assert(!strcmp(tests[i][1], res));
        free(res);
    }
}

void
test_ext() {
    char* tests[][2] = {
        {"libraries/opencl/matmul.foo", "foo"},
        {"blah.MOO", "MOO"},
        {"mooo", ""}
    };
    for (int i = 0; i < ARRAY_SIZE(tests); i++) {
        const char *res = paths_ext(tests[i][0]);
        assert(!strcmp(tests[i][1], res));
    }
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_basename);
    PRINT_RUN(test_dirname);
    PRINT_RUN(test_stem);
    PRINT_RUN(test_ext);
}
