// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef PRETTY_H
#define PRETTY_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
    size_t indent;
    size_t indent_width;
    size_t key_width;
} pretty_printer;

pretty_printer *
pp_init();

void
pp_free(pretty_printer *me);

void
pp_print_key_value(pretty_printer *me,
                   char *key,
                   char *value_fmt, ...);


#endif
