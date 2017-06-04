#include <assert.h>
#include <stdio.h>
#include "isect/isect.h"
#include "linalg/linalg.h"

void
test_moeller_trumbore() {
    vec3 orig = {
        24.492475509643554688,
        24.006366729736328125,
        22.174991607666015625
    };
    vec3 dir = {
        -0.582438647747039795,
        -0.430847525596618652,
        -0.689300775527954102
    };
    vec3 v0 = {
        2.079962015151977539,
        8.324080467224121094,
        -4.233458995819091797
    };
    vec3 v1 = {
        1.942253947257995605,
        8.138879776000976562,
        -3.293735027313232422
    };
    vec3 v2 = {
        2.189547061920166016,
        7.210639953613281250,
        -4.343578815460205078
    };
    float t, u, v;
    assert(isect_moeller_trumbore(orig, dir, v0, v1, v2, &t, &u, &v));
}

void
test_precomp12() {
    vec3 v0 = {2.189, 5.870, 0.517};
    vec3 v1 = {1.795, 4.835, 0.481};
    vec3 v2 = {2.717, 6.016, -1.116};

    float T[12];
    isect_precomp12_precompute(v0, v1, v2, T);

    float Texp[12] = {
        0.000000000000000000,
        -0.963188350200653076,
        -0.086114756762981415,
        5.698436737060546875,
        0.000000000000000000,
        0.021233797073364258,
        -0.610471367835998535,
        0.190971285104751587,
        1.000000000000000000,
        -0.390707582235336304,
        0.288399755954742432,
        -0.044649176299571991
    };
    for (int i = 0; i < 12; i++) {
        assert(approx_eq(T[i], Texp[i]));
    }
}

void
test_diffs_01() {
    float t, u, v;
    vec3 v0 = {1.470782, 7.976924, 3.797243};
    vec3 v1 = {0.767229, 7.976924, 3.966874};
    vec3 v2 = {0.777313, 7.939148, 4.027555};

    vec3 o = {11.998573, 14.635927, 9.681089};
    vec3 d = {-0.891354, -0.050203, -0.450520};

    float T[12];
    isect_precomp12_precompute(v0, v1, v2, T);

    bool mt = isect_moeller_trumbore(o, d, v0, v1, v2, &t, &u, &v);
    bool pc = isect_precomp12(o, d, v0, v1, v2, &t, &u, &v, T);
    assert(mt == pc);
}

void
test_diffs_02() {
    float t, u, v;

    vec3 v0 = {63.000000, 44.000000, 95.000000};
    vec3 v1  = {-46.000000, 26.000000, 57.000000};
    vec3 v2 = {82.000000, 46.000000, -54.000000};

    vec3 o = {0};
    vec3 d = {-0.628539, -0.769961, -0.109994};

    float T[12];
    isect_precomp12_precompute(v0, v1, v2, T);

    bool mt = isect_moeller_trumbore(o, d, v0, v1, v2, &t, &u, &v);
    bool pc = isect_precomp12(o, d, v0, v1, v2, &t, &u, &v, T);
    assert(mt == pc);
}


void
test_diffs_hard() {
    float t, u, v;
    vec3 vec[3];
    vec3 o = {0}, d;
    float T[12];
    d.x = rand_n(200) - 100;
    d.y = rand_n(200) - 100;
    d.z = rand_n(200) - 100;
    d = v3_normalize(d);
    for (int n = 0; n < 100; n++) {
        for (int i = 0; i < 3; i++) {
            vec[i].x = rand_n(200) - 100;
            vec[i].y = rand_n(200) - 100;
            vec[i].z = rand_n(200) - 100;
        }
        isect_precomp12_precompute(vec[0], vec[1], vec[2], T);
        bool mt = isect_moeller_trumbore(o, d, vec[0], vec[1], vec[2], &t, &u, &v);
        bool pc = isect_precomp12(o, d, vec[0], vec[1], vec[2], &t, &u, &v, T);

        if (mt  != pc) {
            for (int i  = 0; i < 3; i++) {
                v3_print(vec[i], 6);
            }
        }
        assert(mt == pc);
    }
}


int
main(int argc, char* argv[]) {
    rand_init(0);
    PRINT_RUN(test_moeller_trumbore);
    PRINT_RUN(test_precomp12);
    PRINT_RUN(test_diffs_01);
    PRINT_RUN(test_diffs_02);
    PRINT_RUN(test_diffs_hard);
    return 0;
}
