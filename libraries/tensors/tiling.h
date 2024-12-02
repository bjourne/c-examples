// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_TILING_H
#define TENSORS_TILING_H

#include "tensors.h"

// Not sure about these names.
void tensor_linearize_tiles(tensor *src, tensor *dst,
                            int tile_height, int tile_width);
tensor *
tensor_linearize_tiles_new2(
    tensor *src,
    unsigned int tile_height, unsigned int tile_width,
    unsigned int fill_height, unsigned int fill_width
);

tensor *
tensor_tile_2d_new(tensor *src, int tile_y, int tile_x);



#endif
