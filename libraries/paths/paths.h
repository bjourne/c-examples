// Copyright (C) 2019, 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef PATHS_H
#define PATHS_H

// Does not allocate memory
const char *paths_basename(const char *path);
const char *paths_ext(const char *path);

// Allocates memory
char *paths_stem(const char *path);
char *paths_dirname(char *path);
char *paths_normalize(const char *path);

#endif
