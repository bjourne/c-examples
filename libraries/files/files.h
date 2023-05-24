// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef FILES_H
#define FILES_H

#include <stdbool.h>

// Perhaps will contain more functions for file handling in the
// future.
bool files_read(const char *path, char **buf, size_t *size);

#endif
