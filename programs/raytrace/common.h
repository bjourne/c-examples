#ifndef COMMON_H
#define COMMON_H

#include "isect/isect.h"

#define ISECT_MT        1
#define ISECT_MT_B      2
#define ISECT_BW9       3
#define ISECT_BW9_B     4
#define ISECT_BW12      5
#define ISECT_BW12_B    6
#define ISECT_SF01      7
#define ISECT_DS        8
#define ISECT_SHEV      9
#define ISECT_HH       10

#define PLAIN_SHADING   1
#define FANCY_SHADING   2

#define ISECT_PC_P ISECT_METHOD == ISECT_BW9 ||                         \
        ISECT_METHOD == ISECT_BW12 ||                                   \
        ISECT_METHOD == ISECT_BW12_B ||                                 \
        ISECT_METHOD == ISECT_BW9_B ||                                  \
        ISECT_METHOD == ISECT_HH ||                                     \
        ISECT_METHOD == ISECT_SHEV

#if ISECT_METHOD == ISECT_MT
    // It should be Möller, not Moller. But printf misaligns text with ö
    // in it.
    #define ISECT_FUN isect_mt
    #define ISECT_NAME "mt"
    #define ISECT_RUNNING_MIN false
#elif ISECT_METHOD == ISECT_MT_B
    #define ISECT_FUN isect_mt_b
    #define ISECT_NAME "mt-b"
    #define ISECT_RUNNING_MIN false
#elif ISECT_METHOD == ISECT_BW9
    #define ISECT_FUN isect_bw9
    #define ISECT_FUN_PRE isect_bw9_pre
    #define ISECT_NAME "bw9"
    #define ISECT_DATA isect_bw9_data
    #define ISECT_RUNNING_MIN true
#elif ISECT_METHOD == ISECT_BW9_B
    #define ISECT_FUN isect_bw9_b
    #define ISECT_FUN_PRE isect_bw9_pre
    #define ISECT_NAME "bw9-b"
    #define ISECT_DATA isect_bw9_data
    #define ISECT_RUNNING_MIN true
#elif ISECT_METHOD == ISECT_BW12
    #define ISECT_FUN isect_bw12
    #define ISECT_FUN_PRE isect_bw12_pre
    #define ISECT_NAME "bw12"
    #define ISECT_DATA isect_bw12_data
    #define ISECT_RUNNING_MIN true
#elif ISECT_METHOD == ISECT_BW12_B
    #define ISECT_FUN isect_bw12_b
    #define ISECT_FUN_PRE isect_bw12_pre
    #define ISECT_NAME "bw12-b"
    #define ISECT_DATA isect_bw12_data
    #define ISECT_RUNNING_MIN true
#elif ISECT_METHOD == ISECT_SF01
    #define ISECT_FUN isect_sf01
    #define ISECT_NAME "sf01"
    #define ISECT_RUNNING_MIN false
#elif ISECT_METHOD == ISECT_DS
    #define ISECT_FUN isect_ds
    #define ISECT_NAME "ds"
    #define ISECT_RUNNING_MIN true
#elif ISECT_METHOD == ISECT_SHEV
    #define ISECT_FUN isect_shev
    #define ISECT_FUN_PRE isect_shev_pre
    #define ISECT_NAME "shev"
    #define ISECT_DATA isect_shev_data
    #define ISECT_RUNNING_MIN false
#elif ISECT_METHOD == ISECT_HH
    #define ISECT_FUN isect_hh
    #define ISECT_FUN_PRE isect_hh_pre
    #define ISECT_NAME "hh"
    #define ISECT_DATA isect_hh_data
    #define ISECT_RUNNING_MIN false
#else
    #error "Wrong ISECT_METHOD"
#endif

typedef struct _ray_intersection {
    float t;
    vec2 uv;
    int tri_idx;
} ray_intersection;

#endif
