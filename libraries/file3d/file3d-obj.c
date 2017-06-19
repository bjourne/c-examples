#include <ctype.h>
#include <string.h>

#include "datatypes/vector.h"
#include "file3d/file3d.h"

typedef union {
    int i;
    float f;
    ptr p;
} int_or_float;

static bool
str_is_empty(const char *s) {
    while (*s != '\0') {
        if (!isspace(*s))
            return false;
        s++;
    }
    return true;
}

static int *
index_array_pack(vector *a, int n_vecs) {
    int *pack = (int *)malloc(sizeof(int) * a->used);
    for (int i = 0; i < a->used; i++) {
        int idx = (int)a->array[i];
        if (idx < 0) {
            idx = n_vecs + idx;
        }
        pack[i] = idx;
    }
    return pack;
}

static vec3 *
v3_array_pack(vector *a) {
    int n = a->used / 3;
    vec3 *out = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        out[i].x = ((int_or_float)a->array[3 * i]).f;
        out[i].y = ((int_or_float)a->array[3 * i + 1]).f;
        out[i].z = ((int_or_float)a->array[3 * i + 2]).f;
    }
    return out;
}

static bool
str_to_tri_indices(char *buf,
                   vector *vertex_indices,
                   vector *normal_indices) {
    int i0, i1, i2;
    int ni0, ni1, ni2;
    int ti0, ti1, ti2;
    if (sscanf(buf, "f %d %d %d", &i0, &i1, &i2) == 3) {
        v_add(vertex_indices, (ptr)(i0 - 1));
        v_add(vertex_indices, (ptr)(i1 - 1));
        v_add(vertex_indices, (ptr)(i2 - 1));
        return true;
    }
    if (sscanf(buf, "f %d//%d %d//%d %d//%d",
               &i0, &ni0,
               &i1, &ni1,
               &i2, &ni2) == 6) {
        v_add(vertex_indices, (ptr)(i0 - 1));
        v_add(vertex_indices, (ptr)(i1 - 1));
        v_add(vertex_indices, (ptr)(i2 - 1));
        v_add(normal_indices, (ptr)(ni0 - 1));
        v_add(normal_indices, (ptr)(ni1 - 1));
        v_add(normal_indices, (ptr)(ni2 - 1));
        return true;
    }
    if (sscanf(buf, "f %d/%d %d/%d %d/%d",
               &i0, &ti0,
               &i1, &ti1,
               &i2, &ti2) == 6) {
        v_add(vertex_indices, (ptr)(i0 - 1));
        v_add(vertex_indices, (ptr)(i1 - 1));
        v_add(vertex_indices, (ptr)(i2 - 1));
        return true;
    }
    return false;
}

static bool
str_to_vertex(char *buf, const char *fmt, vector *a) {
    int_or_float x, y, z;
    if (sscanf(buf, fmt, &x.f, &y.f, &z.f) != 3) {
        return false;
    }
    v_add(a, x.i);
    v_add(a, y.i);
    v_add(a, z.i);
    return true;
}

bool
f3d_load_obj(file3d *me, FILE *f) {
    vector *verts = v_init(10);
    vector *normals = v_init(10);
    vector *vertex_indices = v_init(10);
    vector *normal_indices = v_init(10);
    char buf[1024];
    bool ret = false;
    while (fgets(buf, 1024, f)) {
        if (str_is_empty(buf)) {
            continue;
        }
        if (!strncmp(buf, "v ", 2)) {
            if (!str_to_vertex(buf, "v %f %f %f", verts)) {
                goto end;
            }
        } else if (!strncmp(buf, "vn ", 3)) {
            if (!str_to_vertex(buf, "vn %f %f %f", normals)) {
                goto end;
            }
        } else if (!strncmp(buf, "#", 1)) {
            // Skip comments
        } else if (!strncmp(buf, "mtllib ", 7)) {
            // Skip material lib
        } else if (!strncmp(buf, "usemtl ", 7)) {
            // Skip material use
        } else if (!strncmp(buf, "g ", 2)) {
            // Skip group names
        } else if (!strncmp(buf, "s ", 2)) {
            // Skip smooth shading
        } else if (!strncmp(buf, "vt ", 2)) {
            // Skip texture coords
        } else if (!strncmp(buf, "f ", 2)) {
            if (!str_to_tri_indices(buf,
                                    vertex_indices,
                                    normal_indices)) {
                goto end;
            }
        } else {
            f3d_set_error(me,
                          FILE3D_ERR_OBJ_LINE_UNPARSABLE,
                          buf);
            goto end;
        }
    }
    me->n_tris = vertex_indices->used / 3;

    me->n_verts = verts->used / 3;
    me->verts = v3_array_pack(verts);
    me->vertex_indices = index_array_pack(vertex_indices, me->n_verts);

    me->n_normals = normals->used / 3;
    me->normals = v3_array_pack(normals);
    me->normal_indices = index_array_pack(normal_indices, me->n_normals);
    ret = true;
 end:
    v_free(verts);
    v_free(vertex_indices);
    v_free(normals);
    v_free(normal_indices);
    return ret;
}
