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

    // Automatically break long lines
    char *value = "cl_khr_icd cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_3d_image_writes cl_intel_exec_by_local_thread cl_khr_spir cl_khr_fp64 cl_khr_image2d_from_buffer cl_intel_vec_len_hint";

    pp_print_key_value(pp, "Extensions", "%s", value);
    pp_free(pp);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_key_values);
}
