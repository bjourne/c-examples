// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// Conventions:
//
//   * `n` or `n_els` is the number of elements in the tensor.
//   * Tensors have `int` number of elements.
//   * `eps` means epsilon.
//
// For convolutions and similar code:
//
//   * x, y, and c are the image width, height, and number of channels.
//   * s, d, and f are the source and destination image and f the filter bank (4d).
//   * The suffix _dim is the exclusive maximum.
//
// So dc in [0, dc_dim) represents the current destination channel.
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"
#include "pretty/pretty.h"
#include "random/random.h"
#include "tensors.h"

////////////////////////////////////////////////////////////////////////
// Checking
////////////////////////////////////////////////////////////////////////
void
tensor_check_dims(tensor *t, int n_dims, ...) {
    va_list ap;
    va_start(ap, n_dims);
    assert(t->n_dims == n_dims);
    for (int i = 0; i < n_dims; i++) {
        int d1 = va_arg(ap, int);
        int d2 = t->dims[i];
        if (d1 != d2) {
            printf("Mismatch at %d: %d != %d\n", i, d1, d2);
            assert(false);
        }
    }
    va_end(ap);
}

void
tensor_check_equal_contents(tensor *t1, tensor *t2, float eps) {
    int n_els1 = tensor_n_elements(t1);
    int n_els2 = tensor_n_elements(t2);
    assert(n_els1 == n_els2);
    int n_dims = t1->n_dims;
    int dim_counts[TENSOR_MAX_N_DIMS] = {0};
    int n_mismatches = 0;
    for (int i = 0; i < n_els1; i++) {
        float v1 = t1->data[i];
        float v2 = t2->data[i];
        float diff = fabs(v2 - v1);
        if (diff >= eps) {
            n_mismatches++;
            if (n_mismatches < 100) {
                printf("Mismatch at ");
                tensor_dims_print(n_dims, dim_counts);
                printf(", %10.6f != %10.6f\n", v1,  v2);
            }
        }
        for (int j = n_dims - 1; j >= 0; j--) {
            dim_counts[j]++;
            if (dim_counts[j] == t1->dims[j]) {
                dim_counts[j] = 0;
            } else {
                break;
            }
        }
    }
    if (n_mismatches > 0) {
        printf("%d mismatches in total.\n", n_mismatches);
    }
}

bool
tensor_check_equal(tensor *t1, tensor *t2, float eps) {
    tensor_dims_check_equal(t1->n_dims, t1->dims, t2->n_dims, t2->dims);
    tensor_check_equal_contents(t1, t2, eps);
    return true;
}

////////////////////////////////////////////////////////////////////////
// Printing
////////////////////////////////////////////////////////////////////////
void
tensor_print(tensor *me,
             bool print_header,
             int n_decimals, int n_columns, char *sep) {
    if (print_header) {
        printf("Dims: ");
        tensor_dims_print(me->n_dims, me->dims);
        printf("\n");
    }
    pretty_printer *pp = pp_init();
    pp->n_decimals = n_decimals;
    pp->n_columns = n_columns;
    pp->sep = sep;

    size_t n_dims = me->n_dims;
    size_t dims[PP_MAX_N_DIMS];
    for (int i = 0; i < me->n_dims; i++) {
        dims[i] = me->dims[i];
    }
    pp_print_array(pp, 'f', 4, n_dims, dims, me->data);
    pp_free(pp);
}

////////////////////////////////////////////////////////////////////////
// Dimensions
////////////////////////////////////////////////////////////////////////
void tensor_dims_to_string(int n, int *dims, char *buf) {
    strcat(buf, "[");
    char buf2[256];
    for (int i = 0; i < n - 1; i++) {

        sprintf(buf2, "%d, ", dims[i]);
        strcat(buf, buf2);
    }
    sprintf(buf2, "%d", dims[n - 1]);
    strcat(buf, buf2);
    strcat(buf, "]");
}

void tensor_dims_print(int n, int *dims) {
    char buf[256] = {0};
    tensor_dims_to_string(n, dims, buf);
    printf("%s", buf);
}

void
tensor_dims_check_equal(int n_dims1, int *dims1, int n_dims2, int *dims2) {
    assert(n_dims1 == n_dims2);
    for (int i = 0; i < n_dims1; i++) {
        int d1 = dims1[i];
        int d2 = dims2[i];
        if (d1 != d2) {
            printf("Mismatch at dim %d: %d != %d\n", i, d1, d2);
            assert(false);
        }
    }
}


long
tensor_dims_count(int n, int *ptr) {
    long prod = 1;
    while (n) {
        prod *= *ptr;
        ptr++;
        n--;
    }
    return prod;
}

void
tensor_dims_copy(int src_n, int *src_dims, int *dst_n, int *dst_dims) {
    *dst_n = src_n;
    memcpy(dst_dims, src_dims, sizeof(int) * src_n);
}

void
tensor_dims_clone(int src_n, int *src_dims, int *dst_n, int **dst_dims) {
    *dst_n = src_n;
    *dst_dims = malloc(sizeof(int) * src_n);
    tensor_dims_copy(src_n, src_dims, dst_n, *dst_dims);
}


////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////
int
tensor_padded_strided_dim(int s_dim, int f_dim, int pad, int stride) {
    return (s_dim + 2 * pad - f_dim) / stride + 1;
}

long
tensor_n_elements(tensor *me) {
    return tensor_dims_count(me->n_dims, me->dims);
}

void
tensor_flatten(tensor *me, int from) {
    int n_els = tensor_dims_count(me->n_dims - from, me->dims + from);
    me->n_dims = from + 1;
    me->dims[from] = n_els;
}

void
tensor_set_dims(tensor *me, int n_dims, int dims[]) {
    me->n_dims = n_dims;
    memcpy(me->dims, dims, n_dims * sizeof(int));
}

static float *
permute(float *src, float *dst,
        int left, int *dims,
        int *strides) {
    if (!left) {
        *dst++ = *src;
        return dst;
    }
    for (int i = 0; i < *dims; i++) {
        dst = permute(src, dst, left - 1, dims + 1, strides + 1);
        src += *strides;
    }
    return dst;
}

tensor *
tensor_permute_dims_new(tensor *src, int perm[]) {
    int *src_dims = src->dims;
    int n_dims = src->n_dims;
    int dst_dims[TENSOR_MAX_N_DIMS];
    int cnt = 1;

    int src_strides[TENSOR_MAX_N_DIMS];
    int dst_strides[TENSOR_MAX_N_DIMS];
    for (int i = n_dims - 1; i>= 0; i--) {
        src_strides[i] = cnt;
        cnt *= src_dims[i];
    }
    for (int i = 0; i < n_dims; i++) {
        dst_strides[i] = src_strides[perm[i]];
        dst_dims[i] = src_dims[perm[i]];
    }
    tensor *dst = tensor_init(n_dims, dst_dims);
    permute(src->data, dst->data, n_dims, dst_dims, dst_strides);
    return dst;
}

////////////////////////////////////////////////////////////////////////
// Init and Free
////////////////////////////////////////////////////////////////////////
tensor *
tensor_init(int n_dims, int dims[]) {
    tensor *me = (tensor *)malloc(sizeof(tensor));
    tensor_set_dims(me, n_dims, dims);

    size_t n_bytes = sizeof(float) * tensor_n_elements(me);
    me->data = malloc_aligned(TENSOR_ADDRESS_ALIGNMENT, n_bytes);
    if (!me->data) {
        me->error_code = TENSOR_ERR_TOO_BIG;
    }
    return me;
}

tensor *
tensor_init_1d(int x) {
    return tensor_init(1, (int[]){x});
}

tensor *
tensor_init_2d(int x, int y) {
    return tensor_init(2, (int[]){x, y});
}

tensor *
tensor_init_3d(int x, int y, int z) {
    return tensor_init(3, (int[]){x, y, z});
}

tensor *
tensor_init_4d(int x, int y, int z, int w) {
    return tensor_init(4, (int[]){x, y, z, w});
}

tensor *
tensor_init_copy(tensor *orig) {
    tensor *me = tensor_init(orig->n_dims, orig->dims);
    tensor_copy_data(me, orig->data);
    return me;
}

tensor *
tensor_init_from_data(float *data, int n_dims, int dims[])  {
    tensor *me = tensor_init(n_dims, dims);
    tensor_copy_data(me, data);
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
// Copy data
////////////////////////////////////////////////////////////////////////
void tensor_copy_data(tensor *me, void *data) {
    memcpy(me->data, data, sizeof(float) * tensor_n_elements(me));
}


////////////////////////////////////////////////////////////////////////
// Unary ops
////////////////////////////////////////////////////////////////////////
void
tensor_unary(tensor *src, tensor *dst,
             tensor_unary_op op, float scalar) {
    int n = tensor_n_elements(src);
    assert(n == tensor_n_elements(dst));
    if (op == TENSOR_UNARY_OP_MAX) {
        for (int i = 0; i < n; i++) {
            dst->data[i] = MAX(src->data[i], scalar);
        }
    } else if (op == TENSOR_UNARY_OP_SOFTMAX) {
        float tot = 0.0;
        for (int i = 0; i < n; i++) {
            float v = exp(src->data[i]);
            dst->data[i] = v;
            tot += v;
        }
        tensor_unary(dst, dst, TENSOR_UNARY_OP_DIV, tot);

    } else if (op == TENSOR_UNARY_OP_DIV) {
        for (int i = 0; i < n; i++) {
            dst->data[i] = src->data[i] / scalar;
        }
    } else if (op == TENSOR_UNARY_OP_ADD) {
        for (int i = 0; i < n; i++) {
            dst->data[i] = src->data[i] + scalar;
        }
    } else {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////
// Scans
////////////////////////////////////////////////////////////////////////
void tensor_scan(tensor *src, tensor *dst,
                 tensor_binary_op op, bool exclusive, float seed) {
    int n = tensor_n_elements(src);
    assert(n == tensor_n_elements(dst));

    float at0 = exclusive ? seed : 0.0;
    for (int i = 0; i < n; i++) {
        float v = src->data[i];
        float at1;
        if (op == TENSOR_BINARY_OP_ADD) {
            at1 = at0 + v;
        } else if (op == TENSOR_BINARY_OP_MUL) {
            at1 = at0 * v;
        } else {
            assert(false);
        }
        dst->data[i] = exclusive ? at0 : at1;
        at0 = at1;
    }
}

////////////////////////////////////////////////////////////////////////
// Fills
////////////////////////////////////////////////////////////////////////
void
tensor_fill_const(tensor *me, float v) {
    int n = tensor_n_elements(me);
    if (v == 0.0) {
        memset(me->data, 0, sizeof(float) * n);
    } else {
        for (int i = 0; i < n; i++) {
            me->data[i] = v;
        }
    }
}

void
tensor_fill_rand_range(tensor *me, float high) {
    int n = tensor_n_elements(me);
    // Doesn't this ruin the precision?
    rnd_pcg32_rand_uniform_fill_float(me->data, n);
    for (int i = 0; i < n; i++) {
        me->data[i] *= high;
    }
}

void
tensor_fill_range(tensor *me, float start) {
    int n = tensor_n_elements(me);
    for (int i = 0; i < n; i++) {
        me->data[i] = start;
        start += 1.0;
    }
}

#ifdef HAVE_PNG
#include <png.h>

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
                int dst = 3 * (y * width + x) + c;
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
    int n_bytes = sizeof(float) * 3 * height * width;
    me->data = (float *)malloc_aligned(TENSOR_ADDRESS_ALIGNMENT, n_bytes);
    png_bytepp row_pointers = png_get_rows(png, info);

    for (png_uint_32 y = 0; y < height; y++) {
        png_byte *row = row_pointers[y];
        for (png_uint_32 x = 0; x < width; x++) {
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
#endif

////////////////////////////////////////////////////////////////////////
// 2D Max Pooling
////////////////////////////////////////////////////////////////////////
tensor *
tensor_max_pool2d_new(int fy_dim, int fx_dim,
                      int stride, int padding,
                      tensor *src) {
    int sc_dim = src->dims[0];
    int sy_dim = src->dims[1];
    int sx_dim = src->dims[2];
    int dy_dim = tensor_padded_strided_dim(sy_dim, fy_dim, padding, stride);
    int dx_dim = tensor_padded_strided_dim(sx_dim, fx_dim, padding, stride);
    tensor *dst = tensor_init(3, (int[]){sc_dim, dy_dim, dx_dim});
    tensor_max_pool2d(fy_dim, fx_dim, stride, padding, src, dst);
    return dst;
}


void
tensor_max_pool2d(int kernel_h, int kernel_w,
                  int stride, int padding,
                  tensor *src, tensor *dst)  {

    int src_h = src->dims[1];
    int src_w = src->dims[2];
    int src_n_channels = src->dims[0];
    int src_size = src_h * src_w;

    int dst_h = dst->dims[1];
    int dst_w = dst->dims[2];

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
    // Arguably, this is wrong if weight is multi-dimensional
    tensor *dst = tensor_init_1d(weights->dims[0]);
    tensor_linear(weights, bias, src, dst);
    return dst;
}

void
tensor_transpose(tensor *src, tensor *dst) {
    assert(src->n_dims == 2 &&  src->n_dims == dst->n_dims);
    int src_height = src->dims[0];
    int src_width = src->dims[1];
    int dst_height = dst->dims[0];
    int dst_width = dst->dims[1];
    assert(src_height == dst_width);
    assert(src_width == dst_height);
    for (int i = 0; i < src_height; i++) {
        for  (int j = 0; j < src_width; j++) {
            dst->data[j * dst_width + i] = src->data[i * src_width + j];
        }
    }
}

tensor *tensor_transpose_new(tensor *src) {
    tensor *dst = tensor_init_2d(src->dims[1], src->dims[0]);
    tensor_transpose(src, dst);
    return dst;
}
