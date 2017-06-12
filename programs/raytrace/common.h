#ifndef COMMON_H
#define COMMON_H

#define ISECT_MT        1
#define ISECT_MT_B      2
#define ISECT_BW9       3
#define ISECT_BW9_B     4
#define ISECT_BW12      5
#define ISECT_BW12_B    6
#define ISECT_SF01      7
#define ISECT_DS        8
#define ISECT_SHEV      9

#define PLAIN_SHADING   1
#define FANCY_SHADING   2

#define ISECT_PC_P ISECT_METHOD == ISECT_BW9 ||                         \
        ISECT_METHOD == ISECT_BW12 ||                                   \
        ISECT_METHOD == ISECT_BW12_B ||                                 \
        ISECT_METHOD == ISECT_BW9_B || ISECT_METHOD == ISECT_SHEV

#if ISECT_METHOD == ISECT_BW12 ||   \
    ISECT_METHOD == ISECT_BW12_B
#define ISECT_PC_N_ELS 12
#else
#define ISECT_PC_N_ELS 10
#endif

#if ISECT_METHOD == ISECT_MT
#define ISECT_FUN isect_mt
#elif ISECT_METHOD == ISECT_MT_B
#define ISECT_FUN isect_mt_b
#elif ISECT_METHOD == ISECT_BW9
#define ISECT_FUN isect_bw9
#elif ISECT_METHOD == ISECT_BW9_B
#define ISECT_FUN isect_bw_b
#elif ISECT_METHOD == ISECT_BW12
#define ISECT_FUN isect_bw12
#elif ISECT_METHOD == ISECT_BW12_B
#define ISECT_FUN isect_bw12_b
#elif ISECT_METHOD == ISECT_SF01
#define ISECT_FUN isect_sf01
#elif ISECT_METHOD == ISECT_DS
#define ISECT_FUN isect_ds
#elif ISECT_METHOD == ISECT_SHEV
#define ISECT_FUN isect_shev
#else
#error "Wrong ISECT_METHOD"
#endif

inline const char *
isect_name() {
    // It should be Möller, not Moller. But printf misaligns text with
    // ö in it.
#if ISECT_METHOD == ISECT_MT
    return "Moller-Trumbore";
#elif ISECT_METHOD == ISECT_MT_B
    return "Moller-Trumbore B";
#elif ISECT_METHOD == ISECT_BW9
    return "Baldwin-Weber pre9";
#elif ISECT_METHOD == ISECT_BW9_B
    return "Baldwin-Weber pre9 B";
#elif ISECT_METHOD == ISECT_BW12
    return "Baldwin-Weber pre12";
#elif ISECT_METHOD == ISECT_BW12_B
    return "Baldwin-Weber pre12 B";
#elif ISECT_METHOD == ISECT_SF01
    return "Segura-Feito 01";
#elif ISECT_METHOD == ISECT_DS
    return "Dan Sunday";
#elif ISECT_METHOD == ISECT_SHEV
    return "Shevtsov";
#endif
}

typedef struct _ray_intersection {
    float t;
    vec2 uv;
    int tri_idx;
} ray_intersection;

#endif
