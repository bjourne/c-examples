#ifndef COMMON_H
#define COMMON_H

#define ISECT_MT        1
#define ISECT_PC9       2
#define ISECT_PC12      3
#define ISECT_METHOD ISECT_PC12

#define PLAIN_SHADING   1
#define FANCY_SHADING   2

#define SHADING_STYLE FANCY_SHADING

typedef struct _ray_intersection {
    float t;
    vec2 uv;
    int tri_idx;
} ray_intersection;

#endif
