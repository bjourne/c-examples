// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "pretty/pretty.h"
#include "datatypes/common.h"

void
test_key_values() {
    pretty_printer *pp = pp_init();
    int maj = 1;
    int min = 2;
    int patch = 3;
    pp_print_key_value(pp, "Version", "%d.%d.%d", maj, min, patch);

    char *value = "cl_khr_icd cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_3d_image_writes cl_intel_exec_by_local_thread cl_khr_spir cl_khr_fp64 cl_khr_image2d_from_buffer cl_intel_vec_len_hint";

    pp_print_key_value(pp, "Extensions", "%s", value);
    pp_free(pp);
}

static float
mat_5x5_2[5][5] = {
    {0, 1, 2, 2, 0},
    {1, 3, 2, 1, 0},
    {4, 4, 4, 3, 3},
    {3, 0, 1, 3, 1},
    {2, 4, 1, 2, 0}
};

void
test_2d_array() {
    pretty_printer *pp = pp_init();
    pp->sep = ", ";
    pp_print_array(
        pp,
        'f', 4,
        2, (size_t[]){5, 5},
        mat_5x5_2
    );
    printf("\n");

    // Print again, this time indented one step.
    pp->indent++;
    pp_print_array(
        pp,
        'f', 4,
        3, (size_t[]){1, 5, 5},
        mat_5x5_2
    );

    pp_free(pp);
}

void
test_break_lines() {
    pretty_printer *pp = pp_init();
    pp->n_columns = 30;
    pp->indent++;
    pp_print_array(pp, 'f', 4, 1, (size_t[]){25}, mat_5x5_2);
    pp_free(pp);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_key_values);
    PRINT_RUN(test_2d_array);
    PRINT_RUN(test_break_lines);
}
