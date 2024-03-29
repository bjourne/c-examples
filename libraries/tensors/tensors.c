// Copyright (C) 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// Conventions:
//
//   * `n` or `n_els` is the number of elements in the tensor.
//   * Tensors have `size_t` number of elements.
#include <assert.h>
#include <float.h>
#include <math.h>
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

bool
tensor_check_dims(tensor *me, int n_dims, int dims[]) {
    assert(me->n_dims == n_dims);
    for (int i = 0; i < n_dims; i++) {
        assert(me->dims[i] == dims[i]);
    }
    return true;
}

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

bool
tensor_check_equal(tensor *t1, tensor *t2, float epsilon) {
    assert(t1->n_dims == t2->n_dims);
    int n = t1->n_dims;
    int dim_counts[TENSOR_MAX_N_DIMS] = {0};
    int *dims = t1->dims;
    int n_mismatches = 0;
    size_t n_els = tensor_n_elements(t1);
    for (size_t i = 0; i < n_els; i++) {
        float v1 = t1->data[i];
        float v2 = t2->data[i];
        float diff = fabs(v2 - v1);
        if (diff >= epsilon) {
            n_mismatches++;
            if (n_mismatches < 100) {
                printf("Mismatch at ");
                print_dims(n, dim_counts);
                printf(", %10.6f != %10.6f\n", v1,  v2);
            }
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
    if (n_mismatches > 0) {
        printf("%d mismatches in total.\n", n_mismatches);
    }
    return true;
}

////////////////////////////////////////////////////////////////////////
// Printing
////////////////////////////////////////////////////////////////////////
void
tensor_print(tensor *me,
             bool print_header,
             size_t n_decimals, size_t n_columns, char *sep) {
    if (print_header) {
        printf("Dims: ");
        print_dims(me->n_dims, me->dims);
        printf("\n");
    }
    pretty_printer *pp = pp_init();
    pp->n_decimals = n_decimals;
    pp->n_columns = n_columns;
    pp->sep = sep;

    // Should use size_t everywhere
    size_t n_dims = me->n_dims;
    size_t dims[PP_MAX_N_DIMS];
    for (int i = 0; i < me->n_dims; i++) {
        dims[i] = me->dims[i];
    }
    pp_print_array(pp,
                   'f', 4,
                   n_dims, dims,
                   me->data);
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

static void
compute_2d_dims(tensor *src,
                int kernel_h, int kernel_w,
                int stride, int padding,
                int *height, int *width) {
    *height = (src->dims[1] + 2 * padding - kernel_h) / stride + 1;
    *width = (src->dims[2] + 2 * padding - kernel_w) / stride + 1;
}

static size_t
count_elements_from(int n_dims, int *dims, int from) {
    size_t tot = dims[from];
    for (int i = from + 1; i < n_dims; i++) {
        tot *= dims[i];
    }
    return tot;
}

size_t
tensor_n_elements(tensor *me) {
    return count_elements_from(me->n_dims, me->dims, 0);
}

void
tensor_flatten(tensor *me, int from) {
    int n_els = count_elements_from(me->n_dims, me->dims, from);
    me->n_dims = from + 1;
    me->dims[from] = n_els;
}

////////////////////////////////////////////////////////////////////////
// Init and Free
////////////////////////////////////////////////////////////////////////
tensor *
tensor_init(int n_dims, int dims[]) {
    tensor *me = (tensor *)malloc(sizeof(tensor));
    me->n_dims = n_dims;
    memcpy(me->dims, dims, n_dims * sizeof(int));

    size_t n_bytes = sizeof(float) * tensor_n_elements(me);
    me->data = malloc_aligned(TENSOR_ADDRESS_ALIGNMENT, n_bytes);
    if (!me->data) {
        me->error_code = TENSOR_ERR_TOO_BIG;
    }
    return me;
}

tensor *
tensor_init_copy(tensor *orig) {
    tensor *me = tensor_init(orig->n_dims, orig->dims);
    int n_bytes = sizeof(float) * tensor_n_elements(me);
    memcpy(me->data, orig->data, n_bytes);
    return me;
}

tensor *
tensor_init_from_data(float *data, int n_dims, int dims[])  {
    tensor *me = tensor_init(n_dims, dims);
    memcpy(me->data, data, sizeof(float) * tensor_n_elements(me));
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
// Unary ops
////////////////////////////////////////////////////////////////////////
void
tensor_unary(tensor *src, tensor *dst,
             tensor_unary_op op, float scalar) {
    size_t n = tensor_n_elements(src);
    assert(n == tensor_n_elements(dst));
    if (op == TENSOR_UNARY_OP_MAX) {
        for (size_t i = 0; i < n; i++) {
            dst->data[i] = MAX(src->data[i], scalar);
        }
    } else if (op == TENSOR_UNARY_OP_SOFTMAX) {
        float tot = 0.0;
        for (size_t i = 0; i < n; i++) {
            float v = exp(src->data[i]);
            dst->data[i] = v;
            tot += v;
        }
        tensor_unary(dst, dst, TENSOR_UNARY_OP_DIV, tot);

    } else if (op == TENSOR_UNARY_OP_DIV) {
        for (size_t i = 0; i < n; i++) {
            dst->data[i] = src->data[i] / scalar;
        }
    } else if (op == TENSOR_UNARY_OP_ADD) {
        for (size_t i = 0; i < n; i++) {
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
    size_t n = tensor_n_elements(src);
    assert(n == tensor_n_elements(dst));

    float at0 = exclusive ? seed : 0.0;
    for (size_t i = 0; i < n; i++) {
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
    size_t n = tensor_n_elements(me);
    if (v == 0.0) {
        memset(me->data, 0, sizeof(float) * n);
    } else {
        for (size_t i = 0; i < n; i++) {
            me->data[i] = v;
        }
    }
}

void
tensor_fill_rand_range(tensor *me, float high) {
    size_t n = tensor_n_elements(me);
    // Doesn't this ruin the precision?
    rnd_pcg32_rand_uniform_fill_float(me->data, n);
    for (size_t i = 0; i < n; i++) {
        me->data[i] *= high;
    }
}

void
tensor_fill_range(tensor *me, float start) {
    size_t n = tensor_n_elements(me);
    for (size_t i = 0; i < n; i++) {
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
    size_t n_bytes = sizeof(float) * 3 * height * width;
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
    int height, width;
    compute_2d_dims(src, weight->dims[2], weight->dims[3], stride, padding,
                    &height, &width);
    tensor *dst = tensor_init(3, (int[]){weight->dims[0], height, width});
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
tensor_max_pool2d_new(int kernel_h, int kernel_w,
                      int stride, int padding,
                      tensor *src) {

    int height, width;
    compute_2d_dims(src, kernel_h, kernel_w, stride, padding,
                    &height, &width);
    tensor *dst = tensor_init(3, (int[]){src->dims[0], height, width});
    tensor_max_pool2d(kernel_h, kernel_w, stride, padding, src, dst);
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
    tensor *dst = tensor_init(1, (int[]){weights->dims[0]});
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

    tensor *weight = tensor_init(2, (int[]){out, in});
    tensor *bias = tensor_init(1, (int[]){out});

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

    tensor *weight = tensor_init(4, (int[]){out_chans, in_chans,
                                            kernel_size, kernel_size});
    tensor *bias = tensor_init(1, (int[]){out_chans});
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
    memcpy(weight->data, weight_data, tensor_n_elements(weight) * sizeof(float));
    memcpy(bias->data, bias_data, tensor_n_elements(bias) * sizeof(float));
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
    size_t buf_size = count_elements_from(input_n_dims, input_dims, 0);
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
    me->src_buf = tensor_init(1, (int[]){buf_size});
    me->dst_buf = tensor_init(1, (int[]){buf_size});
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
    tensor_check_dims(input, me->input_n_dims, me->input_dims);
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
