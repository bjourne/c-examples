// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Thin wrapper for CUDA.
#ifndef CUDA_H
#define CUDA_H

#define CUD_ASSERT(err) cud_assert(err, __FILE__, __LINE__)

void
cud_print_system_details();

#endif
