// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_DCT_H
#define TENSORS_DCT_H

#include "tensors/tensors.h"

void tensor_dct2d_rect(tensor *src, tensor *dst,
                       int sy, int sx, int height, int width);
void tensor_dct2d_blocked(tensor *src, tensor *dst,
                          int block_height, int block_width);
void tensor_idct2d(tensor *src, tensor *dst);

#endif
