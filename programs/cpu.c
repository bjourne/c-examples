// Copyright (C) 2019 Björn Lindqvist
// Program for dumping cpu info.
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Very much inspired by https://github.com/Mysticial/FeatureDetector
#ifdef _WIN32

//  Windows
#define cpuid(info, x)    __cpuidex(info, x, 0)

#else

//  GCC Intrinsics
#include <cpuid.h>
void cpuid(int info[4], int type){
    __cpuid_count(type, 0, info[0], info[1], info[2], info[3]);
}

#endif

typedef enum {
    FLAG_MMX = 0,

    FLAG_SSE,
    FLAG_SSE2,
    FLAG_SSE3,
    FLAG_SSSE3,
    FLAG_SSE41,
    FLAG_SSE42,

    FLAG_POPCNT,
    FLAG_AES,
    FLAG_AVX,
    FLAG_AVX2,
    FLAG_FMA3,
    FLAG_RDRAND,
    FLAG_COUNT
} cpu_flags;

static char* flag_names[FLAG_COUNT] = {
    "MMX",
    "SSE", "SSE2", "SSE3", "SSSE3", "SSE41", "SSE42",
    "POPCNT", "AES", "AVX", "AVX2", "FMA3", "RDRAND"
};

typedef struct {
    char vendor[13];
    bool flags[FLAG_COUNT];
} cpu_info;

cpu_info *
cpu_info_get() {
    cpu_info *me = (cpu_info *)malloc(sizeof(cpu_info));

    // info[2] = ECX, info[3] = EDX
    int info[4];

    // Vendor string
    cpuid(info, 0);
    memcpy(me->vendor, &info[1], 4);
    memcpy(me->vendor + 4, &info[3], 4);
    memcpy(me->vendor + 8, &info[2], 4);
    me->vendor[12] = '\0';

    cpuid(info, 0);
    int n_ids = info[0];
    if (n_ids >= 1) {
        cpuid(info, 1);
        me->flags[FLAG_MMX] = (info[3] & ((int)1 << 23)) != 0;
        me->flags[FLAG_SSE] = (info[3] & ((int)1 << 25)) != 0;
        me->flags[FLAG_SSE2] = (info[3] & ((int)1 << 26)) != 0;
        me->flags[FLAG_SSE3] = (info[2] & ((int)1 << 0)) != 0;

        me->flags[FLAG_SSSE3] = (info[2] & ((int)1 <<  9)) != 0;
        me->flags[FLAG_SSE41] = (info[2] & ((int)1 << 19)) != 0;
        me->flags[FLAG_SSE42] = (info[2] & ((int)1 << 20)) != 0;

        me->flags[FLAG_POPCNT] = (info[2] & ((int)1 << 23)) != 0;
        me->flags[FLAG_AES] = (info[2] & ((int)1 << 25)) != 0;
        me->flags[FLAG_AVX] = (info[2] & ((int)1 << 28)) != 0;
        me->flags[FLAG_FMA3] = (info[2] & ((int)1 << 12)) != 0;
        me->flags[FLAG_RDRAND] = (info[2] & ((int)1 << 30)) != 0;
    }
    if (n_ids >= 7) {
        cpuid(info, 7);
        me->flags[FLAG_AVX2] = (info[1] & ((int)1 << 5)) != 0;
    }
    return me;
}

void
cpu_info_print(cpu_info *me) {
    printf("CPU Vendor: %s\n", me->vendor);
    printf("Flags     : ");
    for (int i = 0; i < FLAG_COUNT; i++) {
        if (me->flags[i]) {
            printf("%s ", flag_names[i]);
        }
    }
    printf("\n");
}


int
main(int argc, char *argv[]) {
    cpu_info *info = cpu_info_get();
    cpu_info_print(info);
    free(info);
    return 0;
}
