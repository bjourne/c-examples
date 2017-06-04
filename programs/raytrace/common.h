#ifndef COMMON_H
#define COMMON_H

//#define ISECT_MT
#define ISECT_PRECOMP12

//#define FANCY_SHADING
#define PLAIN_SHADING

typedef struct _ray_intersection {
    float t;
    vec2 uv;
    int tri_idx;
} ray_intersection;

#endif
