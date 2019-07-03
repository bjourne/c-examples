// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <assert.h>
#include "datatypes/common.h"
#include "file3d/file3d.h"

// The models referenced here originally came from
// http://graphics.cs.williams.edu/data/meshes.xml. That site has
// disappeared from the internet and I don't know how to find the
// anymore... They only exist on my harddrive.
#define FILE_COW_GEO "/tmp/cow.geo"
#define FILE_BUNNY_OBJ "/tmp/bunny.obj"
#define FILE_BUNNY_2_OBJ "/tmp/bunny_2.obj"
#define FILE_TEAPOT_GEO "/tmp/teapot.geo"
#define FILE_HEAD_OBJ "/tmp/head.OBJ"

void
test_errors() {
    file3d *f = f3d_load("i dont exist");
    assert(f->error_code == FILE3D_ERR_FILE_NOT_FOUND);
    f3d_free(f);

    f = f3d_load("/etc/passwd");
    assert(f->error_code == FILE3D_ERR_UNKNOWN_EXTENSION);
    f3d_free(f);
}

void
test_geo() {
    file3d *f = f3d_load(FILE_COW_GEO);
    assert(f->error_code == FILE3D_ERR_NONE);
    assert(f->n_tris == 9468 / 3);
    assert(f->n_verts == 1732);
    assert(f->vertex_indices[0] == 2);
    f3d_free(f);

    f = f3d_load(FILE_TEAPOT_GEO);
    assert(f->error_code == FILE3D_ERR_NONE);
    assert(f->n_normals == 18960);
    f3d_free(f);
}

void
test_obj() {
    file3d *f = f3d_load(FILE_BUNNY_OBJ);
    assert(f->error_code == FILE3D_ERR_NONE);
    assert(f->n_tris == 4968);
    f3d_free(f);

    f = f3d_load(FILE_BUNNY_2_OBJ);
    assert(f->error_code == FILE3D_ERR_NONE);
    assert(approx_eq(f->verts[0].x, 0.1102022f));
    assert(approx_eq(f->verts[0].y, 0.74011f));
    assert(approx_eq(f->verts[0].z, 1.132398f));
    f3d_free(f);

    f = f3d_load(FILE_HEAD_OBJ);
    assert(f->error_code == FILE3D_ERR_NONE);
    assert(f->n_tris == 17684);
    assert(f->n_coords == 35368);
    f3d_free(f);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(test_errors);
    PRINT_RUN(test_geo);
    PRINT_RUN(test_obj);
}
