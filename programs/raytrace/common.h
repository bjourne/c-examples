#ifndef COMMON_H
#define COMMON_H

#define ISECT_MT        1
#define ISECT_PC9       2
#define ISECT_PC12      3
#define ISECT_SF01      4
#define ISECT_METHOD ISECT_MT

#define PLAIN_SHADING   1
#define FANCY_SHADING   2

#define SHADING_STYLE FANCY_SHADING

inline const char *
isect_name() {
#if ISECT_METHOD == ISECT_MT
    return "Möller-Trumbore";
#elif ISECT_METHOD == ISECT_PC9
    return "Baldwin-Weber pre9";
#elif ISECT_METHOD == ISECT_PC12
    return "Baldwin-Weber pre12";
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
