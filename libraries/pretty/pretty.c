// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include "pretty/pretty.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

pretty_printer *
pp_init() {
    pretty_printer *me = malloc(sizeof(pretty_printer));
    me->indent = 0;
    me->indent_width = 2;
    me->key_width = 15;
    return me;
}

void
pp_free(pretty_printer *me) {
    free(me);
}

static void
pp_print_prefix(pretty_printer *me) {
    for (size_t i = 0; i < me->indent * me->indent_width; i++) {
        printf(" ");
    }
}

void
pp_print_key_value(
    pretty_printer *me,
    char *key,
    char *value_fmt, ...) {

    char buf[2048];
    sprintf(buf, "%%-%lds: ", me->key_width);
    pp_print_prefix(me);
    printf(buf, key);

    va_list ap;
    va_start(ap, value_fmt);
    vsprintf(buf, value_fmt, ap);
    va_end(ap);
    printf("%s\n", buf);
}
