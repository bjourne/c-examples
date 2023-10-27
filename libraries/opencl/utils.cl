// Copyright (C) 2023 Björn A. Lindqvist <bjourne@gmail.com>
//
// This file is to be included by OpenCL kernels.
#ifndef UTILS_H
#define UTILS_H

#if VECTOR_WIDTH==2
#define VLOAD(o, a) vload2((o), (a))
#define VLOAD_AT(o, a) vload2((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore2((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore2((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VLONG  convert_long2
#define CONVERT_VUINT  convert_uint2
#define CONVERT_VDOUBLE convert_double2
#define CONVERT_VFLOAT convert_float2
typedef uchar2 vuchar;
typedef int2 vint;
typedef uint2 vuint;
typedef long2 vlong;
typedef float2 vfloat;
typedef double2 vdouble;
#elif VECTOR_WIDTH==4
#define VLOAD(o, a) vload4((o), (a))
#define VLOAD_AT(o, a) vload4((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore4((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore4((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VLONG  convert_long4
#define CONVERT_VUINT  convert_uint4
#define CONVERT_VDOUBLE convert_double4
#define CONVERT_VFLOAT convert_float4
typedef uchar4 vuchar;
typedef int4 vint;
typedef uint4 vuint;
typedef long4 vlong;
typedef float4 vfloat;
typedef double4 vdouble;
#elif VECTOR_WIDTH==8
#define VLOAD(o, a) vload8((o), (a))
#define VLOAD_AT(o, a) vload8((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore8((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore8((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VLONG  convert_long8
#define CONVERT_VUINT  convert_uint8
#define CONVERT_VDOUBLE convert_double8
#define CONVERT_VFLOAT convert_float8
typedef uchar8 vuchar;
typedef int8 vint;
typedef uint8 vuint;
typedef long8 vlong;
typedef float8 vfloat;
typedef double8 vdouble;
#elif VECTOR_WIDTH==16
#define VLOAD(o, a) vload16((o), (a))
#define VLOAD_AT(o, a) vload16((o) / VECTOR_WIDTH, (a))
#define VSTORE(item, ofs, arr) vstore16((item), (ofs), (arr))
#define VSTORE_AT(e, o, a) vstore16((e), (o) / VECTOR_WIDTH, (a))
#define CONVERT_VLONG  convert_long16
#define CONVERT_VUINT  convert_uint16
#define CONVERT_VDOUBLE convert_double16
#define CONVERT_VFLOAT convert_float16
typedef uchar16 vuchar;
typedef int16 vint;
typedef uint16 vuint;
typedef long16 vlong;
typedef float16 vfloat;
typedef double16 vdouble;
#else
#error "Set VECTOR_WIDTH to 2, 4, 8, or 16 before including this file."
#endif

#define VLOAD_AT_AS_LONG(o, a) CONVERT_VLONG(VLOAD_AT(o, a))
#define VLOAD_AT_AS_DOUBLE(o, a) CONVERT_VDOUBLE(VLOAD_AT(o, a))
#define VSTORE_AT_AS_UINT(e, o, a) VSTORE_AT(CONVERT_VUINT(e), o, a)

#define ALIGN_TO(n, w)  (((n) + (w)) / (w) * w)

typedef struct {
    uint c0, c1, c2;
} ranges;

ranges
slice_work(uint n_items, uint n_workers, uint id, uint width) {
    ranges r;
    uint chunk = ALIGN_TO(n_items / n_workers + 1, width);
    r.c0 = id * chunk;
    uint n = min(r.c0 + chunk, n_items) - r.c0;
    r.c1 = r.c0 + n / width * width;
    r.c2 = r.c1 + n % width;
    return r;
}

#endif
