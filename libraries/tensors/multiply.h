#ifndef TENSORS_MULTIPLY_H
#define TENSORS_MULTIPLY_H

#include "tensors.h"

// Two functions that does the same thing to make it easy for me to
// test out optimizations.
void tensor_multiply_ref(tensor *a, tensor *b, tensor *c);
void tensor_multiply(tensor *a, tensor *b, tensor *c);
void tensor_linearize_tiles(tensor *src, tensor *dst,
                            int tile_height, int tile_width);

#endif
