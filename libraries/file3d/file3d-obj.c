#include <ctype.h>
#include <string.h>

#include "datatypes/bits.h"
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
        } else {
            idx -= 1;
        }
        pack[i] = idx;
    }
    return pack;
}

static vec3 *
v3_array_pack(vector *a) {
    size_t n = a->used / 3;
    vec3 *out = (vec3 *)malloc(sizeof(vec3) * n);
    for (int i = 0; i < n; i++) {
        out[i].x = BW_PTR_TO_FLOAT(a->array[3 * i]);
        out[i].y = BW_PTR_TO_FLOAT(a->array[3 * i + 1]);
        out[i].z = BW_PTR_TO_FLOAT(a->array[3 * i + 2]);
    }
    return out;
}

static vec2 *
v2_array_pack(vector *a) {
    size_t n = a->used / 2;
    vec2 *out = (vec2 *)malloc(sizeof(vec2) * n);
    for (int i  = 0; i < n; i++) {
        out[i].x = BW_PTR_TO_FLOAT(a->array[2 * i]);
        out[i].y = BW_PTR_TO_FLOAT(a->array[2 * i + 1]);
    }
    return out;
}

static void
add_triple(vector *a, int i0, int i1, int i2) {
    v_add(a, (ptr)i0);
    v_add(a, (ptr)i1);
    v_add(a, (ptr)i2);
}

static bool
str_to_tri_indices(char *buf,
                   vector *vertex_indices,
                   vector *normal_indices,
                   vector *coord_indices) {
    int vi0, vi1, vi2, vi3;
    int ni0, ni1, ni2;
    int ti0, ti1, ti2, ti3;

    // This approach is perhaps a little lazy. :)
    if (sscanf(buf, "f %d %d %d %d", &vi0, &vi1, &vi2, &vi3) == 4) {
        add_triple(vertex_indices, vi0, vi1, vi2);
        add_triple(vertex_indices, vi0, vi2, vi3);
        return true;
    }
    if (sscanf(buf, "f %d/%d %d/%d %d/%d %d/%d",
               &vi0, &ti0,
               &vi1, &ti1,
               &vi2, &ti2,
               &vi3, &ti3) == 8) {
        add_triple(vertex_indices, vi0, vi1, vi2);
        add_triple(coord_indices, ti0, ti1, ti2);
        add_triple(vertex_indices, vi0, vi2, vi3);
        add_triple(coord_indices, ti0, ti2, ti3);
        return true;
    }
    if (sscanf(buf, "f %d %d %d", &vi0, &vi1, &vi2) == 3) {
        add_triple(vertex_indices, vi0, vi1, vi2);
        return true;
    }
    if (sscanf(buf, "f %d//%d %d//%d %d//%d",
               &vi0, &ni0,
               &vi1, &ni1,
               &vi2, &ni2) == 6) {
        add_triple(vertex_indices, vi0, vi1, vi2);
        add_triple(normal_indices, ni0, ni1, ni2);
        return true;
    }
    if (sscanf(buf, "f %d/%d %d/%d %d/%d",
               &vi0, &ti0,
               &vi1, &ti1,
               &vi2, &ti2) == 6) {
        add_triple(vertex_indices, vi0, vi1, vi2);
        add_triple(coord_indices, ti0, ti1, ti2);
        return true;
    }
    if (sscanf(buf, "f %d/%d/%d %d/%d/%d %d/%d/%d",
               &vi0, &ti0, &ni0,
               &vi1, &ti1, &ni1,
               &vi2, &ti2, &ni2) == 9) {
        add_triple(vertex_indices, vi0, vi1, vi2);
        add_triple(normal_indices, ni0, ni1, ni2);
        add_triple(coord_indices, ti0, ti1, ti2);
        return true;
    }
    return false;
}

static bool
str_to_v3(char *buf, const char *fmt, vector *a) {
    int_or_float x, y, z;
    if (sscanf(buf, fmt, &x.f, &y.f, &z.f) != 3) {
        return false;
    }
    v_add(a, x.i);
    v_add(a, y.i);
    v_add(a, z.i);
    return true;
}

static bool
str_to_v2(char *buf, char *fmt, vector *a) {
    int_or_float x, y;
    if (sscanf(buf, fmt, &x.f, &y.f) != 2) {
        return false;
    }
    v_add(a, x.i);
    v_add(a, y.i);
    return true;
}

void
f3d_load_obj(file3d *me, FILE *f) {
    vector *verts = v_init(10);
    vector *normals = v_init(10);
    vector *coords = v_init(10);
    vector *vertex_indices = v_init(10);
    vector *normal_indices = v_init(10);
    vector *coord_indices = v_init(10);
    char buf[1024];
    while (fgets(buf, 1024, f)) {
        if (str_is_empty(buf)) {
            continue;
        }
        if (!strncmp(buf, "v ", 2)) {
            if (!str_to_v3(buf, "v %f %f %f", verts)) {
                goto end;
            }
        } else if (!strncmp(buf, "vn ", 3)) {
            if (!str_to_v3(buf, "vn %f %f %f", normals)) {
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
            if (!str_to_v2(buf, "vt %f %f", coords)) {
                goto end;
            }
        } else if (!strncmp(buf, "f ", 2)) {
            if (!str_to_tri_indices(buf,
                                    vertex_indices,
                                    normal_indices,
                                    coord_indices)) {
                goto end;
            }
        } else {
            f3d_set_error(me,
                          FILE3D_ERR_OBJ_LINE_UNPARSABLE,
                          buf);
            goto end;
        }
    }

    int n_v_indices = (int)vertex_indices->used;
    int n_n_indices = (int)normal_indices->used;
    int n_c_indices = (int)coord_indices->used;
    if ((n_n_indices > 0 && n_n_indices != n_v_indices) ||
        (n_c_indices > 0 && n_c_indices != n_v_indices)) {
        f3d_set_error(me, FILE3D_ERR_OBJ_FACE_VARYING, NULL);
        goto end;
    }

    me->n_tris = n_v_indices / 3;
    me->n_verts = (int)verts->used / 3;
    me->verts = v3_array_pack(verts);
    me->vertex_indices = index_array_pack(vertex_indices, me->n_verts);

    me->n_normals = (int)normals->used / 3;
    me->normals = v3_array_pack(normals);
    me->normal_indices = index_array_pack(normal_indices, me->n_normals);

    me->n_coords = (int)coords->used / 2;
    me->coords = v2_array_pack(coords);
    me->coord_indices = index_array_pack(coord_indices, me->n_coords);
 end:
    v_free(verts);
    v_free(vertex_indices);
    v_free(normals);
    v_free(normal_indices);
    v_free(coords);
    v_free(coord_indices);
}
