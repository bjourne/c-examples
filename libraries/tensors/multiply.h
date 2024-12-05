// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_MULTIPLY_H
#define TENSORS_MULTIPLY_H

#include "tensors.h"

// Simple and slow matrix multiplication, used to iron out bugs in the
// optimized implementation.
void tensor_multiply_ref(tensor *a, tensor *b, tensor *c);

// Quite fast multi-threaded matrix multiplication, utilizing SIMD
// intrinsics.
void tensor_multiply_w_params(tensor *a, tensor *b, tensor *c, int n_jobs);
void tensor_multiply(tensor *a, tensor *b, tensor *c);
tensor *tensor_multiply_new(tensor *a, tensor *b);


#endif
