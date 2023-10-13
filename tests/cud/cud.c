// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "cud/cud.h"
#include "datatypes/common.h"

void
test_cud() {
    cud_print_system_details();
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_cud);
}
