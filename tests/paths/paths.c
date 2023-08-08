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
    for (size_t i = 0; i < ARRAY_SIZE(tests); i++) {
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
    for (size_t i = 0; i < ARRAY_SIZE(tests); i++) {
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
    for (size_t i = 0; i < ARRAY_SIZE(tests); i++) {
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
    for (size_t i = 0; i < ARRAY_SIZE(tests); i++) {
        const char *res = paths_ext(tests[i][0]);
        assert(!strcmp(tests[i][1], res));
    }
}

void
test_normalize() {
    char* tests[][2] = {
        {"", "."},
        {"./.", "."},
        {"///", "/"},
        {"///.", "/"},
        {".././.", ".."},
        {"foo" , "foo"},
        {"/foo", "/foo"},
        {"foo/.", "foo"},
        {"///foo", "/foo"},
        {"foo/bar/baz", "foo/bar/baz"},
        {"foo/bar//baz//", "foo/bar/baz"},
        {"eeh/", "eeh"},
        {"/", "/"},
        {"foo/./bar", "foo/bar"},
        {"./foo", "foo"},
        {"/./.", "/"}
    };
    for (size_t i = 0; i < ARRAY_SIZE(tests); i++) {
        char *inp = tests[i][0];
        char *exp = tests[i][1];
        char *res = paths_normalize(inp);
        printf("%-20s => %-20s, expected: %-20s\n", inp, res, exp);
        assert(!strcmp(exp, res));
        free(res);
    }
}

void
test_join() {
    char* tests[][3] = {
        {"foo", "/bar", "/bar"},
        {"foo/", "bar", "foo/bar"},
        {"foo", "bar", "foo/bar"},
        {"", "", "."},
        {"..///.q", ".", "../.q"},
        {"/foo", "bar/", "/foo/bar"},
        {"0123456789abcdef", "12345678", "0123456789abcdef/12345678"}
    };
    for (size_t i = 0; i < ARRAY_SIZE(tests); i++) {
        char *x = tests[i][0];
        char *y = tests[i][1];
        char *exp = tests[i][2];
        char *got = paths_join(x, y);
        printf("%-20s + %-20s => %-20s, got: %-20s\n", x, y, exp, got);
        assert(!strcmp(exp, got));
        free(got);
    }
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_basename);
    PRINT_RUN(test_dirname);
    PRINT_RUN(test_stem);
    PRINT_RUN(test_ext);
    PRINT_RUN(test_normalize);
    PRINT_RUN(test_join);
}
