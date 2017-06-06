#ifndef COMMON_H
#define COMMON_H

#define ISECT_MT        1
#define ISECT_MT_B      2
#define ISECT_PC9       3
#define ISECT_PC12      4
#define ISECT_PC12_B    5
#define ISECT_SF01      6
#define ISECT_METHOD ISECT_PC9

#define PLAIN_SHADING   1
#define FANCY_SHADING   2

#define SHADING_STYLE FANCY_SHADING

#define ISECT_PC_P ISECT_METHOD == ISECT_PC9 || \
        ISECT_METHOD == ISECT_PC12 || ISECT_METHOD == ISECT_PC12_B

#if ISECT_METHOD == ISECT_PC12 || ISECT_METHOD == ISECT_PC12_B
#define ISECT_PC_SIZE 12
#else
#define ISECT_PC_SIZE 10
#endif

inline const char *
isect_name() {
#if ISECT_METHOD == ISECT_MT
    return "Möller-Trumbore";
#elif ISECT_METHOD == ISECT_MT_B
    return "Möller-Trumbore B";
#elif ISECT_METHOD == ISECT_PC9
    return "Baldwin-Weber pre9";
#elif ISECT_METHOD == ISECT_PC12
    return "Baldwin-Weber pre12";
#elif ISECT_METHOD == ISECT_PC12_B
    return "Baldwin-Weber pre12 B";
#elif ISECT_METHOD == ISECT_SF01
    return "Segura-Feito 01";
#endif
}

typedef struct _ray_intersection {
    float t;
    vec2 uv;
    int tri_idx;
} ray_intersection;

#endif
