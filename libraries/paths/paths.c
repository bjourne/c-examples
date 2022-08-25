// Copyright (C) 2019, 2022 Björn Lindqvist <bjourne@gmail.com>
#include <string.h>
#include "datatypes/common.h"
#include "paths.h"

// These functions works almost like
// http://man7.org/linux/man-pages/man3/basename.3.html, except that
// they support backslashes which are used on Windows. On Unix, a
// backslash can be part of a filename so they are technically not
// correct.
const char *
paths_basename(const char *path) {
    char *p1 = strrchr(path, '\\');
    char *p2 = strrchr(path, '/');
    char *p = MAX(p1, p2);
    if (!p) {
        return path;
    }
    return ++p;
}

char *
paths_stem(const char *path) {
    const char *path2 = paths_basename(path);
    char *str = strdup(path2);
    char *p = strchr(str, '.');
    if (p) {
        *p = '\0';
    }
    return str;
}

// There are some weird edge cases which I just ignore here.
char *
paths_dirname(char *path) {
    char *p1 = strrchr(path, '\\');
    char *p2 = strrchr(path, '/');
    char *p = MAX(p1, p2);
    if (!p) {
        return strdup("");
    }
    while ((*p == '/' || *p == '\\') && p != path) {
        p--;
    }
    p++;
    int len = (int)(p - path);
    char *buf = (char *)malloc(len + 1);
    strncpy(buf, path, len);
    buf[len] = '\0';
    return buf;
}
