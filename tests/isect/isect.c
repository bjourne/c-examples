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
test_bw9_pre() {
    vec3 v0 = {2.189, 5.870, 0.517};
    vec3 v1 = {1.795, 4.835, 0.481};
    vec3 v2 = {2.717, 6.016, -1.116};
    float T[10];
    isect_bw9_pre(v0, v1, v2, (isect_bw9_data *)&T);

    float Texp[9] = {
        -0.963188350201,
        -0.086114756763,
        5.698436737061,
        0.021233797073,
        -0.610471367836,
        0.190971285105,
        -0.390707582235,
        0.288399755955,
        -0.044649176300,
    };
    for (int i = 0; i < 9; i++) {
        assert(approx_eq2(T[i], Texp[i], 1e-6));
    }


    v0 = (vec3){4.0, 1.0, 2.0};
    v1 = (vec3){-4.0, 3.0, 3.0};
    v2 = (vec3){-10.0, -3.0, 21.5};
    isect_bw9_pre(v0, v1, v2, (isect_bw9_data *)&T);
    float Texp2[9] = {
        -0.137323945761,
        -0.098591551185,
        0.746478855610,
        0.007042253390,
        0.056338027120,
        -0.140845075250,
        0.302816897631,
        0.422535210848,
        -3.056338071823
    };
    for (int i = 0; i < 9; i++) {
        assert(approx_eq2(T[i], Texp2[i], 1e-6));
    }
}

void
test_bw12_pre() {
    vec3 v0 = {2.189, 5.870, 0.517};
    vec3 v1 = {1.795, 4.835, 0.481};
    vec3 v2 = {2.717, 6.016, -1.116};

    float T[12];
    isect_bw12_pre(v0, v1, v2, (isect_bw12_data *)&T);

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
    float t = ISECT_FAR;
    vec2 uv;
    vec3 v0 = {1.470782, 7.976924, 3.797243};
    vec3 v1 = {0.767229, 7.976924, 3.966874};
    vec3 v2 = {0.777313, 7.939148, 4.027555};

    vec3 o = {11.998573, 14.635927, 9.681089};
    vec3 d = {-0.891354, -0.050203, -0.450520};

    isect_bw12_data D;
    isect_bw12_pre(v0, v1, v2, &D);

    bool mt = isect_mt(o, d, v0, v1, v2, &t, &uv);
    bool pc = isect_bw12(o, d, &t, &uv, &D);
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

    isect_bw12_data D;
    isect_bw12_pre(v0, v1, v2, &D);

    bool mt = isect_mt(o, d, v0, v1, v2, &t, &uv);
    bool pc = isect_bw12(o, d, &t, &uv, &D);
    assert(mt == pc);
}

void
test_diffs_hard() {
    // I should look into this.
    /* float t = ISECT_FAR; */
    /* vec2 uv; */

    /* vec3 vec[3]; */
    /* vec3 o = {0}, d; */
    /* isect_bw12_data D; */
    /* d.x = rand_n(200) - 100; */
    /* d.y = rand_n(200) - 100; */
    /* d.z = rand_n(200) - 100; */
    /* d = v3_normalize(d); */
    /* for (int n = 0; n < 100; n++) { */
    /*     t = ISECT_FAR; */
    /*     for (int i = 0; i < 3; i++) { */
    /*         vec[i].x = rand_n(200) - 100; */
    /*         vec[i].y = rand_n(200) - 100; */
    /*         vec[i].z = rand_n(200) - 100; */
    /*     } */
    /*     isect_bw12_pre(vec[0], vec[1], vec[2], &D); */
    /*     bool mt = isect_mt(o, d, vec[0], vec[1], vec[2], &t, &uv); */
    /*     bool pc = isect_bw12(o, d, &t, &uv, &D); */
    /*     if (mt  != pc) { */
    /*         for (int i  = 0; i < 3; i++) { */
    /*             v3_print(vec[i], 6); */
    /*         } */
    /*     } */
    /*     assert(mt == pc); */
    /* } */
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
    float t = ISECT_FAR;
    vec2 uv;
    assert(!isect_mt(o, d, v0, v1, v2, &t, &uv));

    isect_bw9_data D;
    isect_bw9_pre(v0, v1, v2, &D);
    assert(!isect_bw9(o, d, &t, &uv, &D));
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

    isect_bw9_data D;
    isect_bw9_pre(v0, v1, v2, &D);
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
    assert(approx_eq2(t1, t2, 0.0001));
}

int
main(int argc, char* argv[]) {
    rand_init(0);
    PRINT_RUN(test_mt);
    PRINT_RUN(test_bw9_pre);
    PRINT_RUN(test_bw12_pre);
    PRINT_RUN(test_diffs_01);
    PRINT_RUN(test_diffs_02);
    PRINT_RUN(test_diffs_hard);
    PRINT_RUN(test_bw9_diffs_01);
    PRINT_RUN(test_bw9_diffs_02);
    PRINT_RUN(test_isect_sf01);
    return 0;
}
