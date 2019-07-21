// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
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
        char *res = paths_basename(tests[i][0]);
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

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_basename);
    PRINT_RUN(test_dirname);
}
