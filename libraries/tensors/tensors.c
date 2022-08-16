// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include <float.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include "datatypes/common.h"
#include "tensors.h"

////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////
static void
compute_2d_dims(tensor *src,
                int kernel_h, int kernel_w,
                int stride, int padding,
                int *height, int *width) {
    *height = (src->dims[1] + 2 * padding - kernel_h) / stride + 1;
    *width = (src->dims[2] + 2 * padding - kernel_w) / stride + 1;
}

static int
count_elements_from(tensor *me, int from) {
    int tot = me->dims[from];
    for (int i = from + 1; i < me->n_dims; i++) {
        tot *= me->dims[i];
    }
    return tot;
}

int
tensor_n_elements(tensor *me) {
    int tot = me->dims[0];
    for (int i = 1; i < me->n_dims; i++) {
        tot *= me->dims[i];
    }
    return tot;
}

void
tensor_flatten(tensor *me, int from) {
    int n_els = count_elements_from(me, from);
    me->n_dims = from + 1;
    me->dims[from] = n_els;
}

////////////////////////////////////////////////////////////////////////
// Init and Free
////////////////////////////////////////////////////////////////////////
static tensor *
init_from_va_list(int n_dims, va_list ap) {

    tensor *me = (tensor *)malloc(sizeof(tensor));
    me->n_dims = n_dims;
    for (int i = 0; i < n_dims; i++) {
        me->dims[i] = va_arg(ap, int);
    }
    me->data = (float *)malloc(sizeof(float) * tensor_n_elements(me));
    return me;
}

tensor *
tensor_init(int n_dims, ...) {
    va_list ap;
    va_start(ap, n_dims);
    tensor *me = init_from_va_list(n_dims, ap);
    va_end(ap);
    return me;
}

tensor *
tensor_init_from_data(float *data, int n_dims, ...)  {
    va_list ap;
    va_start(ap, n_dims);
    tensor *me = init_from_va_list(n_dims, ap);
    va_end(ap);
    memcpy(me->data, data, tensor_n_elements(me) *  sizeof(float));
    return me;
}

tensor *
tensor_init_filled(float v, int n_dims, ...)  {
    va_list ap;
    va_start(ap, n_dims);
    tensor *me = init_from_va_list(n_dims, ap);
    va_end(ap);
    tensor_fill(me, v);
    return me;
}

void
tensor_free(tensor *t) {
    if (t->data) {
        free(t->data);
    }
    free(t);
}

////////////////////////////////////////////////////////////////////////
// Scalar Ops
////////////////////////////////////////////////////////////////////////
void
tensor_relu(tensor *me) {
    for (int i  = 0; i < tensor_n_elements(me); i++) {
        me->data[i] = MAX(me->data[i], 0.0f);
    }
}

void
tensor_fill(tensor *me, float v) {
    for (int i = 0; i < tensor_n_elements(me); i++) {
        me->data[i] = v;
    }
}

void
tensor_randrange(tensor *me, int high) {
    for (int i = 0; i < tensor_n_elements(me); i++) {
        me->data[i] = rand_n(high);
    }
}

// Should check dimensions too, but whatever.
bool
tensor_check_equal(tensor *t1, tensor *t2) {
    assert(t1->n_dims == t2->n_dims);
    int n = t1->n_dims;
    int dim_counts[TENSOR_MAX_N_DIMS] = {0};
    int *dims = t1->dims;
    for (int i = 0; i < tensor_n_elements(t1); i++) {
        float v1 = t1->data[i];
        float v2 = t2->data[i];
        if (v1 != v2) {
            printf("Mismatch at [");
            for (int j = 0; j < n - 1; j++) {
                printf("%d, ", dim_counts[j]);
            }
            printf("%d], %.2f != %.2f\n",
                   dim_counts[n - 1], v1,  v2);
        }
        for (int j = n - 1; j >= 0; j--) {
            dim_counts[j]++;
            if (dim_counts[j] == dims[j]) {
                dim_counts[j] = 0;
            } else {
                break;
            }
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////
// PNG Support
////////////////////////////////////////////////////////////////////////
// TODO: Check with valgrind if read/write png leaks.
bool
tensor_write_png(tensor *me, char *filename) {
    if (me->n_dims != 3 || me->dims[0] != 3) {
        me->error_code = TENSOR_ERR_WRONG_DIMENSIONALITY;
        return false;
    }
    int height = me->dims[1];
    int width = me->dims[2];
    FILE *f = NULL;
    png_structp png = NULL;

    f = fopen(filename, "wb");
    if (!f)
        goto error;

    png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                  NULL, NULL, NULL);
    if (!png)
        goto error;

    png_infop info = png_create_info_struct(png);
    if (!info)
        goto error;

    if (setjmp(png_jmpbuf(png)))
        goto error;

    png_init_io(png, f);

    // Write header
    if (setjmp(png_jmpbuf(png)))
        goto error;

    png_set_IHDR(png, info, width, height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png, info);

    // Write bytes
    if (setjmp(png_jmpbuf(png))) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return false;
    }

    png_byte *image = malloc(width * height * 3);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src = c * width * height + y * width + x;
                int dst = 3*(y * width + x) + c;
                image[dst] = me->data[src];
            }
        }
    }

    png_bytep *row_pointers = (png_bytep *)malloc(
        sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = image + (width * 3 * y);
    }
    png_write_image(png, row_pointers);

    // End write
    if (setjmp(png_jmpbuf(png))) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return false;
    }
    png_write_end(png, NULL);
    free(row_pointers);
    free(image);

    png_destroy_write_struct(&png, &info);
    fclose(f);
    me->error_code = TENSOR_ERR_NONE;
    return true;

 error:
    if (f)
        fclose(f);
    if (png)
        png_destroy_write_struct(&png, &info);

    me->error_code = TENSOR_ERR_PNG_ERROR;
    return false;
}

tensor *
tensor_read_png(char *filename) {
    tensor *me = (tensor *)malloc(sizeof(tensor));
    me->data = NULL;
    FILE *f = NULL;
    png_structp png = NULL;


    f = fopen(filename, "rb");
    png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                 NULL, NULL, NULL);
    if (!png) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        goto done;
    }
    png_infop info = png_create_info_struct(png);
    png_init_io(png, f);
    png_read_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    png_uint_32 width, height;
    int depth, ctype, interlace_method, compression_method, filter_method;
    png_get_IHDR(png, info, &width, &height, &depth,
                 &ctype, &interlace_method, & compression_method,
                 &filter_method);
    if  (depth != 8) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        goto done;
    }

    int bpp;
    if (ctype == PNG_COLOR_TYPE_RGB) {
        bpp = 3;
    } else if (ctype == PNG_COLOR_TYPE_RGBA) {
        bpp = 4;
    } else {
        me->error_code = TENSOR_ERR_UNSUPPORTED_PNG_TYPE;
        goto done;
    }

    // Skip alpha channel if present.
    me->dims[0] = 3;
    me->dims[1] = height;
    me->dims[2] = width;
    me->n_dims = 3;
    me->data = (float *)malloc(sizeof(float) * 3 * height * width);
    png_bytepp row_pointers = png_get_rows(png, info);

    for (int y = 0; y < height; y++) {
        png_byte *row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                png_byte v = row[c];
                me->data[c*width*height + y*width + x] = v;
            }
            row += bpp;
        }
    }
    me->error_code = TENSOR_ERR_NONE;
 done:
    if (f)
        fclose(f);
    if (png)
        png_destroy_read_struct(&png, &info, NULL);

    return me;
}

////////////////////////////////////////////////////////////////////////
// 2D Convolution
////////////////////////////////////////////////////////////////////////
tensor *
tensor_conv2d_new(tensor *weight, tensor *bias,
                  int stride, int padding, tensor *src) {
    int height, width;
    compute_2d_dims(src, weight->dims[2], weight->dims[3], stride, padding,
                    &height, &width);
    tensor *dst = tensor_init(3, weight->dims[0], height, width);
    tensor_conv2d(weight, bias, stride, padding, src, dst);
    return dst;
}

void
tensor_conv2d(tensor *weight, tensor *bias,
              int stride, int padding,
              tensor *src, tensor *dst) {

    int weight_c_out = weight->dims[0];
    int weight_c_in = weight->dims[1];
    int weight_h = weight->dims[2];
    int weight_w = weight->dims[3];

    int src_h = src->dims[1];
    int src_w = src->dims[2];

    int h_start = -padding;
    int h_end = src_h + padding - weight_h + 1;
    int w_start = -padding;
    int w_end = src_w + padding - weight_w + 1;

    int dst_h, dst_w;
    compute_2d_dims(src, weight_h, weight_w, stride, padding,
                    &dst_h, &dst_w);

    int dst_size = dst_h * dst_w;

    assert(weight->n_dims == 4);
    assert(bias->n_dims == 1);
    assert(bias->dims[0] == weight_c_out);

    assert(dst->n_dims == 3);
    assert(dst->dims[0] == weight_c_out);
    assert(dst->dims[1] == dst_h);
    assert(dst->dims[2] == dst_w);

    assert(src->n_dims == 3);
    assert(src->dims[0] == weight_c_in);

    int src_size = src_h * src_w;
    int weight_size = weight_w * weight_h;
    for (int c_out = 0; c_out < weight_c_out; c_out++) {
        float *weight_ptr = &weight->data[c_out * weight_c_in * weight_size];
        for (int c_in = 0; c_in < weight_c_in; c_in++) {
            float *dst_ptr = &dst->data[c_out * dst_size];
            float *src_ptr = &src->data[c_in * src_size];
            for (int h = h_start; h < h_end; h += stride) {
                for (int w = w_start; w < w_end; w += stride) {
                    float acc;
                    if (c_in > 0) {
                        acc = *dst_ptr;
                    } else {
                        acc = bias->data[c_out];
                    }
                    float *weight_ptr2 = &weight_ptr[c_in * weight_size];
                    for  (int i3 = 0; i3 < weight_h; i3++) {
                        for (int i4 = 0; i4 < weight_w; i4++)  {
                            int at1 = h + i3;
                            int at2 = w + i4;

                            float s = 0;
                            if (at1 >= 0 && at1 < src_h &&
                                at2 >= 0 && at2 < src_w) {
                                s = src_ptr[at1 * src_w + at2];
                            }
                            float weight = *weight_ptr2;
                            acc += s * weight;
                            weight_ptr2++;
                        }
                    }
                    *dst_ptr = acc;
                    dst_ptr++;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// 2D Max Pooling
////////////////////////////////////////////////////////////////////////
tensor *
tensor_max_pool2d_new(tensor *src,
                      int kernel_h, int kernel_w,
                      int stride, int padding) {
    int height, width;
    compute_2d_dims(src, kernel_h, kernel_w, stride, padding,
                    &height, &width);
    tensor *dst = tensor_init(3, src->dims[0], height, width);
    tensor_max_pool2d(src, kernel_h, kernel_w, dst, stride, padding);
    return dst;
}


void
tensor_max_pool2d(tensor  *src,
                  int kernel_h, int kernel_w,
                  tensor *dst,
                  int stride, int padding)  {

    int src_h = src->dims[1];
    int src_w = src->dims[2];
    int src_n_channels = src->dims[0];
    int src_size = src_h * src_w;

    int dst_h, dst_w;
    compute_2d_dims(src, kernel_h, kernel_w, stride, padding,
                    &dst_h, &dst_w);

    assert(src->n_dims == 3);
    assert(dst->n_dims == 3);
    assert(dst->dims[0] == src_n_channels);
    assert(dst->dims[1] == dst_h);
    assert(dst->dims[2] == dst_w);

    // Iteration boundaries
    int start_y = -padding;
    int end_y = src_h + padding - kernel_h + 1;
    int start_x = -padding;
    int end_x = src_w + padding - kernel_w + 1;

    // Write address
    float *dst_ptr = dst->data;

    for (int c = 0; c < src_n_channels; c++) {
        float *src_ptr = &src->data[c * src_size];
        for (int y = start_y; y < end_y; y += stride) {
            for (int x = start_x; x < end_x; x += stride) {
                float max = -FLT_MAX;
                for (int ky = 0; ky < kernel_h; ky++) {
                    for (int kx = 0; kx < kernel_w; kx++) {
                        int at_y = y + ky;
                        int at_x = x + kx;
                        if (at_y >= 0 && at_y < src_h &&
                            at_x >= 0 && at_x < src_w) {
                            float s = src_ptr[at_y * src_w + at_x];
                            if (s > max) {
                                max = s;
                            }
                        }
                    }
                }
                *dst_ptr = max;
                dst_ptr++;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Linear
////////////////////////////////////////////////////////////////////////
void
tensor_linear(tensor *weights, tensor *bias, tensor *src, tensor *dst) {
    int n_dims = weights->n_dims;
    int width = weights->dims[n_dims - 1];
    int height = weights->dims[n_dims - 2];
    for (int y = 0; y < height; y++) {
        float acc = 0.0f;
        for (int x = 0; x < width; x++) {
            acc += src->data[x] * weights->data[y * width + x];
        }
        dst->data[y] = acc + bias->data[y];
    }
}

tensor *
tensor_linear_new(tensor *weights, tensor *bias, tensor *src) {
    tensor *dst = tensor_init(1, weights->dims[0]);
    tensor_linear(weights, bias, src, dst);
    return dst;
}

////////////////////////////////////////////////////////////////////////
// Layer abstraction
////////////////////////////////////////////////////////////////////////
tensor_layer *
tensor_layer_init_relu() {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    return me;
}

tensor_layer *
tensor_layer_init_flatten(int from) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->flatten.from = from;
    return me;
}

tensor_layer *
tensor_layer_init_linear(int in, int out) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_LINEAR;

    tensor *weight = tensor_init(2, out, in);
    tensor *bias = tensor_init(1, out);

    me->linear.weight = weight;
    me->linear.bias = bias;

    return me;
}

tensor_layer *
tensor_layer_init_max_pool_2d(int kernel_height, int kernel_width) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_MAX_POOL2D;

    me->max_pool2d.kernel_width = kernel_width;
    me->max_pool2d.kernel_height = kernel_height;

    return me;
}

tensor_layer *
tensor_layer_init_conv2d(int in_chans, int out_chans,
                         int kernel_size,
                         int stride, int padding) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_CONV2D;

    tensor *weight = tensor_init(4, out_chans, in_chans,
                                 kernel_size, kernel_size);
    tensor *bias = tensor_init(1, out_chans);
    me->conv2d.weight = weight;
    me->conv2d.bias = bias;
    me->conv2d.stride = stride;
    me->conv2d.padding = padding;

    return me;
}

void
tensor_layer_free(tensor_layer *me) {
    if (me->type == TENSOR_LAYER_LINEAR)  {
        tensor_free(me->linear.weight);
        tensor_free(me->linear.bias);
    }  else if (me->type == TENSOR_LAYER_CONV2D) {
        tensor_free(me->conv2d.weight);
        tensor_free(me->conv2d.bias);
    }
    free(me);
}
