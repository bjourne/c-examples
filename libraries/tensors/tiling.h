// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_TILING_H
#define TENSORS_TILING_H

#include "tensors.h"

// TODO: delete this
void tensor_linearize_tiles(tensor *src, tensor *dst,
                            int tile_height, int tile_width);


tensor *
tensor_linearize_tiles_new2(
    tensor *src,
    unsigned int tile_y, unsigned int tile_x,
    unsigned int fill_y, unsigned int fill_x
);

// Produces a tiled version of a viewport of size fill_y*fill_x
// overlayed over the matrix. fill_y and fill_x must be multiples of
// tile_y and tile_x, respectively. Elements outside the matrix
// dimensions are treated as zeros. If fill_y or fill_x is zero, then
// the least amount of tiles needed to cover the matrix are used.
tensor *
tensor_tile_2d_new(tensor *src,
                   int tile_y, int tile_x,
                   int fill_y, int fill_x);



#endif
