// Copyright (C) 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// Conventions:
//
//   * `n` or `n_els` is the number of elements in the tensor.
//   * Tensors have `int` number of elements.
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

static void
str_dims(char *buf, int n_dims, int dims[])  {
    strcat(buf, "[");
    char buf2[256];
    for (int i = 0; i < n_dims - 1; i++) {

        sprintf(buf2, "%d, ", dims[i]);
        strcat(buf, buf2);
    }
    sprintf(buf2, "%d", dims[n_dims - 1]);
    strcat(buf, buf2);
    strcat(buf, "]");
}

static void
print_dims(int n_dims, int dims[]) {
    char buf[256] = {0};
    str_dims(buf, n_dims, dims);
    printf("%s", buf);
}

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

static void
check_equal_dims(
    int n_dims1, int dims1[],
    int n_dims2, int dims2[]
) {
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
                print_dims(n_dims, dim_counts);
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
tensor_check_equal(tensor *t1, tensor *t2, float epsilon) {
    check_equal_dims(t1->n_dims, t1->dims,
                     t2->n_dims, t2->dims);
    tensor_check_equal_contents(t1, t2, epsilon);
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
        print_dims(me->n_dims, me->dims);
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
// Utility
////////////////////////////////////////////////////////////////////////
static void
copy_dims(int src_n_dims, int *src_dims,
          int *dst_n_dims, int *dst_dims) {
    *dst_n_dims = src_n_dims;
    memcpy(dst_dims, src_dims, sizeof(int) * TENSOR_MAX_N_DIMS);
}


int
tensor_padded_strided_dim(int s_dim, int f_dim, int pad, int stride) {
    return (s_dim + 2 * pad - f_dim) / stride + 1;
}

static long
count_elements_from(int n_dims, int *dims, int from) {
    long tot = dims[from];
    for (int i = from + 1; i < n_dims; i++) {
        tot *= dims[i];
    }
    return tot;
}

long
tensor_n_elements(tensor *me) {
    return count_elements_from(me->n_dims, me->dims, 0);
}

void
tensor_flatten(tensor *me, int from) {
    int n_els = count_elements_from(me->n_dims, me->dims, from);
    me->n_dims = from + 1;
    me->dims[from] = n_els;
}

void
tensor_set_dims(tensor *me, int n_dims, int dims[]) {
    me->n_dims = n_dims;
    memcpy(me->dims, dims, n_dims * sizeof(int));
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
// 2D Convolution
////////////////////////////////////////////////////////////////////////
tensor *
tensor_conv2d_new(tensor *weight, tensor *bias,
                  int stride, int padding, tensor *src) {

    int sy_dim = src->dims[1];
    int sx_dim = src->dims[2];
    int fy_dim = weight->dims[2];
    int fx_dim = weight->dims[3];

    int dc_dim = weight->dims[0];
    int dy_dim = tensor_padded_strided_dim(sy_dim, fy_dim, padding, stride);
    int dx_dim = tensor_padded_strided_dim(sx_dim, fx_dim, padding, stride);

    tensor *dst = tensor_init(3, (int[]){dc_dim, dy_dim, dx_dim});
    tensor_conv2d(weight, bias, stride, padding, src, dst);
    return dst;
}

#define ADDR3D(d0, d1, d2, i0, i1, i2) (i0*d1*d2 + i1*d2 + i2)
#define ADDR4D(d0, d1, d2, d3, i0, i1, i2, i3) \
    (i0*d1*d2*d3 + i1*d2*d3 + i2*d3 + i3)

#define ADDR5D(d0, d1, d2, d3, d4, i0, i1, i2, i3, i4) \
    (i0*d1*d2*d3*d4 + i1*d2*d3*d4 + i2*d3*d4 + i3*d4 + i4)

void
tensor_conv2d(tensor *weight, tensor *bias,
              int stride, int padding,
              tensor *src, tensor *dst) {

    int fy_dim = weight->dims[2];
    int fx_dim = weight->dims[3];

    int sc_dim = src->dims[0];
    int sy_dim = src->dims[1];
    int sx_dim = src->dims[2];

    int dc_dim = dst->dims[0];
    int dy_dim = dst->dims[1];
    int dx_dim = dst->dims[2];

    tensor_check_dims(weight, 4, dc_dim, sc_dim, fy_dim, fx_dim);
    tensor_check_dims(bias, 1, dc_dim);
    tensor_check_dims(dst, 3, dc_dim, dy_dim, dx_dim);

    assert(src->n_dims == 3);

    float *F = weight->data;
    float *B = bias->data;
    float *S = src->data;
    float *D = dst->data;

    for (int dc = 0; dc < dc_dim; dc++) {
        for (int dy = 0; dy < dy_dim; dy++) {
            for (int dx = 0; dx < dx_dim; dx++) {
                int d_addr = ADDR3D(dc_dim, dy_dim, dx_dim, dc, dy, dx);
                float acc = B[dc];
                for (int sc = 0; sc < sc_dim; sc++) {
                    for (int fy = 0; fy < fy_dim; fy++) {
                        for (int fx = 0; fx < fx_dim; fx++)  {
                            int ay = stride*dy + fy - padding;
                            int ax = stride*dx + fx - padding;
                            float s = 0;
                            int s_addr = ADDR3D(sc_dim, sy_dim, sx_dim, sc, ay, ax);
                            int f_addr = ADDR4D(dc_dim, sc_dim, fy_dim, fx_dim,
                                                dc, sc, fy, fx);
                            if (ay >= 0 && ay < sy_dim &&
                                ax >= 0 && ax < sx_dim) {
                                s = S[s_addr];
                            }
                            float weight = F[f_addr];
                            acc += s * weight;
                        }
                    }
                }
                D[d_addr] = acc;
            }
        }
    }
}

// As usual, dimensions have to be correct
void
tensor_im2col(
    tensor *src, tensor *dst,
    int stride_y, int stride_x,
    int pad_y, int pad_x
) {
    assert(src->n_dims == 3);
    int sy_dim = src->dims[0];
    int sx_dim = src->dims[1];
    int sc_dim = src->dims[2];

    assert(dst->n_dims == 5);
    int dy_dim = dst->dims[0];
    int dx_dim = dst->dims[1];
    int fy_dim = dst->dims[2];
    int fx_dim = dst->dims[3];
    assert(dst->dims[4] == sc_dim);


    float *D = dst->data;
    float *S = src->data;
    for (int dy = 0; dy < dy_dim; dy++) {
        for (int dx = 0; dx < dx_dim; dx++) {
            for (int fy = 0; fy < fy_dim; fy++) {
                for (int fx = 0; fx < fx_dim; fx++) {
                    int sy = stride_y*dy + fy - pad_y;
                    int sx = stride_x*dx + fx - pad_x;
                    /* if (sy >= 0 && sy < sy_dim && sx >= 0 && sx < sx_dim) { */
                    /*     float *at = &S[ADDR3D(sy_dim, sx_dim, sc_dim, sy, sx, 0)]; */
                    /*     for (int sc = 0; sc < sc_dim; sc++) { */
                    /*         *D++ = *at++; */
                    /*     } */
                    /* } else { */
                    /*     for (int sc = 0; sc < sc_dim; sc++) { */
                    /*         *D++ = 0; */
                    /*     } */
                    /* } */
                    for (int sc = 0; sc < sc_dim; sc++) {
                        float v = 0;
                        if (sy >= 0 && sy < sy_dim && sx >= 0 && sx < sx_dim) {
                            int s_addr = ADDR3D(sy_dim, sx_dim, sc_dim, sy, sx, sc);
                            v = S[s_addr];
                        }
                        *D++ = v;
                    }
                }
            }
        }
    }
}

tensor *
tensor_im2col_new(
    tensor *src,
    int fy_dim, int fx_dim,
    int stride_y, int stride_x,
    int pad_y, int pad_x
) {
    assert(src->n_dims == 3);
    int sy_dim = src->dims[0];
    int sx_dim = src->dims[1];
    int sc_dim = src->dims[2];

    int dy_dim = tensor_padded_strided_dim(sy_dim, fy_dim, pad_y, stride_y);
    int dx_dim = tensor_padded_strided_dim(sx_dim, fx_dim, pad_x, stride_x);

    int d_dims[] = {dy_dim, dx_dim, fy_dim, fx_dim, sc_dim};
    tensor *dst = tensor_init(5, d_dims);

    tensor_im2col(src, dst, stride_y, stride_x, pad_y, pad_x);
    return dst;
}

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
    // This is not right?
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


////////////////////////////////////////////////////////////////////////
// Layer abstraction
////////////////////////////////////////////////////////////////////////
tensor_layer *
tensor_layer_init_relu() {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_RELU;
    return me;
}

tensor_layer *
tensor_layer_init_flatten(int from) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_FLATTEN;
    me->flatten.from = from;
    return me;
}

tensor_layer *
tensor_layer_init_linear(int in, int out) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_LINEAR;

    tensor *weight = tensor_init_2d(out, in);
    tensor *bias = tensor_init_1d(out);

    me->linear.weight = weight;
    me->linear.bias = bias;

    return me;
}

tensor_layer *
tensor_layer_init_max_pool2d(int kernel_height, int kernel_width,
                             int stride, int padding) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_MAX_POOL2D;
    me->max_pool2d.kernel_width = kernel_width;
    me->max_pool2d.kernel_height = kernel_height;
    me->max_pool2d.stride = stride;
    me->max_pool2d.padding = padding;
    return me;
}

tensor_layer *
tensor_layer_init_conv2d(int in_chans, int out_chans,
                         int kernel_size,
                         int stride, int padding) {
    tensor_layer *me = (tensor_layer *)malloc(sizeof(tensor_layer));
    me->type = TENSOR_LAYER_CONV2D;

    tensor *weight = tensor_init_4d(out_chans, in_chans,
                                    kernel_size, kernel_size);
    tensor *bias = tensor_init_1d(out_chans);
    me->conv2d.weight = weight;
    me->conv2d.bias = bias;
    me->conv2d.stride = stride;
    me->conv2d.padding = padding;
    return me;
}

tensor_layer *
tensor_layer_init_conv2d_from_data(int in_chans, int out_chans,
                                   int kernel_size,
                                   int stride, int padding,
                                   float *weight_data, float *bias_data) {
    tensor_layer *l = tensor_layer_init_conv2d(in_chans, out_chans,
                                               kernel_size,
                                               stride, padding);
    tensor *weight = l->conv2d.weight;
    tensor *bias = l->conv2d.bias;

    tensor_copy_data(weight, weight_data);
    tensor_copy_data(bias, bias_data);
    return l;
}

int
tensor_layer_n_params(tensor_layer *me) {
    tensor_layer_type t = me->type;
    if (t == TENSOR_LAYER_CONV2D || t == TENSOR_LAYER_LINEAR) {
        tensor *w, *b;
        if (t == TENSOR_LAYER_CONV2D) {
            w = me->conv2d.weight;
            b = me->conv2d.bias;
        } else {
            w = me->linear.weight;
            b = me->linear.bias;
        }
        return tensor_n_elements(w) + tensor_n_elements(b);
    }
    return 0;
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

tensor *
tensor_layer_apply_new(tensor_layer *me, tensor *input) {
    tensor_layer_type t = me->type;
    if (t == TENSOR_LAYER_LINEAR) {
        return tensor_linear_new(me->linear.weight, me->linear.bias, input);
    } else if (t == TENSOR_LAYER_CONV2D) {
        return tensor_conv2d_new(me->conv2d.weight, me->conv2d.bias,
                                 me->conv2d.stride, me->conv2d.padding,
                                 input);
    } else if (t == TENSOR_LAYER_RELU) {
        tensor *output = tensor_init_copy(input);
        tensor_unary(output, output, TENSOR_UNARY_OP_MAX, 0);
        return output;
    } else if (t == TENSOR_LAYER_MAX_POOL2D) {
        return tensor_max_pool2d_new(me->max_pool2d.kernel_width,
                                     me->max_pool2d.kernel_height,
                                     me->max_pool2d.stride,
                                     me->max_pool2d.padding, input);
    } else if (t == TENSOR_LAYER_FLATTEN) {
        tensor *output = tensor_init_copy(input);
        tensor_flatten(output, me->flatten.from);
        return output;
    } else {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////
// Stack abstraction
////////////////////////////////////////////////////////////////////////
tensor_layer_stack *
tensor_layer_stack_init(int n_layers, tensor_layer **layers,
                        int input_n_dims, int *input_dims) {

    tensor *input = tensor_init(input_n_dims, input_dims);

    tensor_layer_stack *me = (tensor_layer_stack *)
        malloc(sizeof(tensor_layer_stack));
    me->n_layers = n_layers;
    me->layers = layers;

    int n_bytes_dims = sizeof(int) * TENSOR_MAX_N_DIMS;

    copy_dims(input->n_dims, input->dims, &me->input_n_dims, me->input_dims);

    // Layers' dims
    me->layers_n_dims = (int *)malloc(sizeof(int) * n_layers);
    me->layers_dims = (int **)malloc(sizeof(int *) * n_layers);
    int buf_size = count_elements_from(input_n_dims, input_dims, 0);
    for (int i = 0; i < n_layers; i++) {
        tensor *output = tensor_layer_apply_new(me->layers[i], input);
        int n_dims = output->n_dims;
        int *dims = output->dims;
        buf_size = MAX(buf_size, count_elements_from(n_dims, dims, 0));
        me->layers_dims[i] = (int *)malloc(n_bytes_dims);
        copy_dims(n_dims, dims, &me->layers_n_dims[i], me->layers_dims[i]);
        tensor_free(input);
        input = output;
    }
    tensor_free(input);
    me->src_buf = tensor_init_1d(buf_size);
    me->dst_buf = tensor_init_1d(buf_size);
    return me;
}

void
tensor_layer_stack_free(tensor_layer_stack *me) {
    for (int i = 0; i < me->n_layers; i++) {
        tensor_layer_free(me->layers[i]);
        free(me->layers_dims[i]);
    }
    tensor_free(me->src_buf);
    tensor_free(me->dst_buf);
    free(me->layers_dims);
    free(me->layers_n_dims);
    free(me);
}

// The point is to avoid redundant mallocs and copies.
tensor *
tensor_layer_stack_apply_new(tensor_layer_stack *me, tensor *input) {
    check_equal_dims(
        input->n_dims, input->dims,
        me->input_n_dims, me->input_dims
    );
    tensor *src = me->src_buf;
    tensor *dst = me->dst_buf;
    copy_dims(input->n_dims, input->dims, &src->n_dims, src->dims);
    memcpy(src->data, input->data, tensor_n_elements(src) * sizeof(float));

    for (int i = 0; i < me->n_layers; i++) {
        tensor_layer *l = me->layers[i];
        tensor_layer_type t = l->type;

        copy_dims(me->layers_n_dims[i], me->layers_dims[i],
                  &dst->n_dims, dst->dims);

        // Content is in dst after
        bool swap = false;
        if (t == TENSOR_LAYER_RELU) {
            tensor_unary(src, src, TENSOR_UNARY_OP_MAX, 0);
        } else if (t == TENSOR_LAYER_CONV2D) {
            tensor_conv2d(l->conv2d.weight, l->conv2d.bias,
                          l->conv2d.stride, l->conv2d.padding,
                          src, dst);
            swap = true;
        } else if (t == TENSOR_LAYER_MAX_POOL2D) {
            tensor_max_pool2d(l->max_pool2d.kernel_width,
                              l->max_pool2d.kernel_height,
                              l->max_pool2d.stride,
                              l->max_pool2d.padding,
                              src, dst);
            swap = true;
        } else if (t == TENSOR_LAYER_FLATTEN) {
            tensor_flatten(src, l->flatten.from);
        } else if (t == TENSOR_LAYER_LINEAR) {
            tensor_linear(l->linear.weight, l->linear.bias,
                          src, dst);
            swap = true;
        } else {
            assert(false);
        }
        if (swap) {
            tensor *tmp = src;
            src = dst;
            dst = tmp;
        }
    }
    return tensor_init_copy(src);
}

static const char *
layer_name(tensor_layer_type t) {
    if (t == TENSOR_LAYER_LINEAR) {
        return "Linear";
    } else if (t == TENSOR_LAYER_RELU) {
        return "ReLU";
    } else if (t == TENSOR_LAYER_MAX_POOL2D) {
        return "MaxPool2D";
    } else if (t == TENSOR_LAYER_CONV2D) {
        return "Conv2D";
    } else if (t == TENSOR_LAYER_FLATTEN)  {
        return "Flatten";
    }
    return "Unknown";
}

static void
layer_details(char *buf, tensor_layer *l) {
    tensor_layer_type t = l->type;
    if (t == TENSOR_LAYER_CONV2D || t == TENSOR_LAYER_LINEAR) {
        tensor *w, *b;
        if (t == TENSOR_LAYER_CONV2D) {
            w = l->conv2d.weight;
            b = l->conv2d.bias;
        } else {
            w = l->linear.weight;
            b = l->linear.bias;
        }
        str_dims(buf, w->n_dims, w->dims);
        strcat(buf, ", ");
        str_dims(buf, b->n_dims, b->dims);
    }
}

void
tensor_layer_stack_print(tensor_layer_stack *me) {
    int n_layers = me->n_layers;
    int n_params = 0;
    for (int i = 0; i < n_layers; i++) {
        n_params += tensor_layer_n_params(me->layers[i]);
    }
    printf("Input: ");
    print_dims(me->input_n_dims, me->input_dims);
    printf(", Layers: %d, Params: %d\n", n_layers, n_params);
    for (int i = 0; i < n_layers; i++) {
        char buf[256] =  {0};
        tensor_layer *l = me->layers[i];
        tensor_layer_type t = l->type;
        layer_details(buf, l);
        printf("  %-10s %-20s: ", layer_name(t), buf);
        print_dims(me->layers_n_dims[i], me->layers_dims[i]);
        printf("\n");
    }
}
