#include "intersection.h"

extern inline bool
moeller_trumbore_isect(vec3 orig, vec3 dir,
                       vec3 v0, vec3 v1, vec3 v2,
                       float *t, float *u, float *v);

extern inline bool
precomp12_isect(vec3 orig, vec3 dir,
                vec3 v0, vec3 v1, vec3 v2,
                float *t, float *u, float *v,
                float *T);
