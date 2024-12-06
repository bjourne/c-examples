// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef TENSORS_CONV2D_H
#define TENSORS_CONV2D_H

#include "tensors.h"

void tensor_conv2d(tensor *weight, tensor *bias,
                   int stride, int padding,
                   tensor *src, tensor *dst);

tensor *tensor_conv2d_new(tensor *weight, tensor *bias,
                          int stride, int padding,
                          tensor *src);
void
tensor_im2col(tensor *src, tensor *dst,
              int stride_y, int stride_x,
              int pad_y, int pad_x);

tensor *
tensor_im2col_new(tensor *src,
                  int fy_dim, int fx_dim,
                  int stride_y, int stride_x,
                  int pad_y, int pad_x);


#endif
