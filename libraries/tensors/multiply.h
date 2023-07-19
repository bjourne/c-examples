// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
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

// Not sure about these names.
void tensor_linearize_tiles(tensor *src, tensor *dst,
                            int tile_height, int tile_width);
tensor *tensor_linearize_tiles_new(tensor *src,
                                   int tile_height,
                                   int tile_width);
tensor *
tensor_linearize_tiles_new2(
    tensor *src,
    unsigned int tile_height, unsigned int tile_width,
    unsigned int fill_height, unsigned int fill_width);

#endif
