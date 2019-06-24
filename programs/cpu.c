// Copyright (C) 2019 Björn Lindqvist
// Program for dumping cpu info.
#include <stdbool.h>
#include <stdio.h>

// https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set
#ifdef _WIN32

//  Windows
#define cpuid(info, x)    __cpuidex(info, x, 0)

#else

//  GCC Intrinsics
#include <cpuid.h>
void cpuid(int info[4], int InfoType){
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}

#endif

int
main(int argc, char *argv[]) {
    int info[4];
    cpuid(info, 0);
    int n_ids = info[0];
    cpuid(info, 0x80000000);
    int n_exts = info[0];
    printf("#ids = %d, #exts = %d\n", n_ids, n_exts);

    cpuid(info,0x00000001);
    bool HW_MMX = (info[3] & ((int)1 << 23)) != 0;
    printf("mmx %d\n", HW_MMX);
    return 0;
}
