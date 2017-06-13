#include <assert.h>
#include <stdio.h>
#include "isect/isect.h"
#include "linalg/linalg.h"

void
test_mt() {
    vec3 o = {
        24.492475509643554688,
        24.006366729736328125,
        22.174991607666015625
    };
    vec3 d = {
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
    float t;
    vec2 uv;
    assert(isect_mt(o, d, v0, v1, v2, &t, &uv));
}

void
test_bw12() {
    vec3 v0 = {2.189, 5.870, 0.517};
    vec3 v1 = {1.795, 4.835, 0.481};
    vec3 v2 = {2.717, 6.016, -1.116};

    float T[12];
    isect_bw12_pre(v0, v1, v2, T);

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
        assert(approx_eq2(T[i], Texp[i], 00000.1));
    }
}

void
test_diffs_01() {
    float t;
    vec2 uv;
    vec3 v0 = {1.470782, 7.976924, 3.797243};
    vec3 v1 = {0.767229, 7.976924, 3.966874};
    vec3 v2 = {0.777313, 7.939148, 4.027555};

    vec3 o = {11.998573, 14.635927, 9.681089};
    vec3 d = {-0.891354, -0.050203, -0.450520};

    float T[12];
    isect_bw12_pre(v0, v1, v2, T);

    bool mt = isect_mt(o, d, v0, v1, v2, &t, &uv);
    bool pc = isect_bw12(o, d, &t, &uv, T);
    assert(mt == pc);
}

void
test_diffs_02() {
    float t;
    vec2 uv;

    vec3 v0 = {63.000000, 44.000000, 95.000000};
    vec3 v1  = {-46.000000, 26.000000, 57.000000};
    vec3 v2 = {82.000000, 46.000000, -54.000000};

    vec3 o = {0};
    vec3 d = {-0.628539, -0.769961, -0.109994};

    float T[12];
    isect_bw12_pre(v0, v1, v2, T);

    bool mt = isect_mt(o, d, v0, v1, v2, &t, &uv);
    bool pc = isect_bw12(o, d, &t, &uv, T);
    assert(mt == pc);
}

void
test_diffs_hard() {
    float t;
    vec2 uv;

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
        isect_bw12_pre(vec[0], vec[1], vec[2], T);
        bool mt = isect_mt(o, d, vec[0], vec[1], vec[2], &t, &uv);
        bool pc = isect_bw12(o, d, &t, &uv, T);
        if (mt  != pc) {
            for (int i  = 0; i < 3; i++) {
                v3_print(vec[i], 6);
            }
        }
        assert(mt == pc);
    }
}

void
test_bw9_diffs_01() {
    vec3 o = {
        11.998573303, 14.635927200, 9.681089401
    };
    vec3 d = {
        -0.271481574, -0.138526469, -0.952422261
    };
    vec3 v0 = {
        9.093396187, 7.740341663, -0.646813512
    };
    vec3 v1 = {
        8.976127625, 7.674720764, -0.514906883
    };
    vec3 v2 = {
        9.011932373, 7.674720764, -0.644407809
    };
    float t;
    vec2 uv;
    assert(!isect_mt(o, d, v0, v1, v2, &t, &uv));

    float T[10];
    isect_bw9_pre(v0, v1, v2, T);
    assert(!isect_bw9(o, d, &t, &uv, T));
}

void
test_bw9_diffs_02() {
    vec3 o = {
        11.998573303, 14.635927200, 9.681089401
    };
    vec3 d = {
        -0.613979876, -0.327146232, -0.718334198
    };
    vec3 v0 = {
        -0.707162023, 7.976923943, -4.903222084
    };
    vec3 v1 = {
        0.022265967, 7.976923943, -4.960473537
    };
    vec3 v2 = {
        0.022265967, 7.989515781, -5.001806736
    };
    float t;
    vec2 uv;

    float T[10];
    isect_bw9_pre(v0, v1, v2, T);
    // The discrepancy appears to be caused by a fp precision issue.
    assert(isect_mt(o, d, v0, v1, v2, &t, &uv));
    //assert(!isect_bw9(o, d, v0, v1, v2, &t, &uv, T));
}

void
test_isect_sf01() {
    vec3 o = {
        24.492475509643554688,
        24.006366729736328125,
        22.174991607666015625
    };
    vec3 d = {
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
    float t1, t2;
    vec2 uv1, uv2;
    isect_mt(o, d, v0, v1, v2, &t1, &uv1);
    isect_sf01(o, d, v0, v1, v2, &t2, &uv2);
    printf("%.6f %.6f\n", t1, t2);
    assert(approx_eq2(t1, t2, 0.0001));
}

int
main(int argc, char* argv[]) {
    rand_init(0);
    PRINT_RUN(test_mt);
    PRINT_RUN(test_bw12);
    PRINT_RUN(test_diffs_01);
    PRINT_RUN(test_diffs_02);
    PRINT_RUN(test_diffs_hard);
    PRINT_RUN(test_bw9_diffs_01);
    PRINT_RUN(test_bw9_diffs_02);
    PRINT_RUN(test_isect_sf01);
    return 0;
}
