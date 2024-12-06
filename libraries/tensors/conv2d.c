#include <assert.h>
#include "conv2d.h"

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
