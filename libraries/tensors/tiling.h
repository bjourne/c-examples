// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_TILING_H
#define TENSORS_TILING_H

#include "tensors.h"

// Produces a tiled version of a viewport of size fill_y*fill_x
// overlayed over the matrix. fill_y and fill_x must be multiples of
// tile_y and tile_x, respectively. Elements outside the matrix
// dimensions are treated as zeros. If fill_y or fill_x is zero, then
// the least amount of tiles needed to cover the matrix are used.
tensor *
tensor_tile_2d_mt_new(tensor *src,
                      int tile_y, int tile_x,
                      int fill_y, int fill_x);

tensor *
tensor_tile_2d_new(tensor *src,
                   int tile_y, int tile_x,
                   int fill_y, int fill_x);

void
tensor_tile_2d(tensor *src, tensor *dst);

void
tensor_transpose_tiled(tensor *src, tensor *dst);



#endif
