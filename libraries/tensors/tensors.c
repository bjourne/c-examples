// Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include "tensors.h"

static int
count_elements(tensor *me) {
    int tot = me->dims[0];
    for (int i = 1; i < me->n_dims; i++) {
        tot *= me->dims[i];
    }
    return tot;
}

tensor *
tensor_allocate(int n_dims, ...) {
    va_list ap;
    va_start(ap, n_dims);

    tensor *me = (tensor *)malloc(sizeof(tensor));
    me->n_dims = n_dims;
    for (int i = 0; i < n_dims; i++) {
        me->dims[i] = va_arg(ap, int);
    }
    me->data = (float *)malloc(sizeof(float) * count_elements(me));
    va_end(ap);
    return me;
}

void
tensor_fill(tensor *me, float v) {
    for (int i = 0; i < count_elements(me); i++) {
        me->data[i] = v;
    }
}

// May be a small leak in this.
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
    FILE *f = fopen(filename, "rb");
    if (!f) {
        me->error_code = TENSOR_ERR_FILE_NOT_FOUND;
        return me;
    }
    unsigned char header[8];
    fread(header, 1, 8, f);
    if (png_sig_cmp(header, 0, 8)) {
        me->error_code = TENSOR_ERR_NOT_A_PNG_FILE;
        return me;
    }
    png_structp png = png_create_read_struct(
        PNG_LIBPNG_VER_STRING,
        NULL, NULL, NULL);
    if (!png) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return me;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return me;
    }

    if (setjmp(png_jmpbuf(png))) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return me;
    }

    png_init_io(png, f);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_byte color_type = png_get_color_type(png, info);
    if  (color_type != PNG_COLOR_TYPE_RGB) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return me;
    }

    png_byte bit_depth = png_get_bit_depth(png, info);
    if  (bit_depth != 8) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return me;
    }
    if (setjmp(png_jmpbuf(png))) {
        me->error_code = TENSOR_ERR_PNG_ERROR;
        return me;
    }

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);

    png_bytep *row_pointers = (png_bytep *)malloc(
        sizeof(png_bytep) * height);


    int bytes_per_row = png_get_rowbytes(png, info);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte *)malloc(bytes_per_row);
    }

    png_read_image(png, row_pointers);

    me->dims[0] = 3;
    me->dims[1] = height;
    me->dims[2] = width;
    me->n_dims = 3;
    me->data = (float *)malloc(sizeof(float) * 3 * height * width);
    for (int y = 0; y < height; y++) {
        png_byte *row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_byte *p = &row[x * 3];
            for (int c = 0; c < 3; c++) {
                int v = p[c];
                me->data[c*width*height + y*width + x] = v;
            }
        }
    }
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    me->error_code = TENSOR_ERR_NONE;

    png_destroy_read_struct(&png, &info, 0);
    fclose(f);
    return me;
}

void
tensor_free(tensor *t) {
    if (t->data) {
        free(t->data);
    }
    free(t);
}
