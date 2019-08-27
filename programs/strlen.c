// Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"

#define BUF_SIZE 1024 * 1024 * 100
#define N_CALLS 100

size_t
optimized_strlen(const char *str) {
    const char *char_ptr;
    const unsigned long int *longword_ptr;
    unsigned long int longword, magic_bits, himagic, lomagic;

    for (char_ptr = str; ((uintptr_t) char_ptr
                          & (sizeof (longword) - 1)) != 0;
         ++char_ptr)
        if (*char_ptr == '\0')
            return char_ptr - str;

    longword_ptr = (unsigned long int *) char_ptr;

    magic_bits = 0x7efefeffL;
    himagic = 0x80808080L;
    lomagic = 0x01010101L;
    if (sizeof (longword) > 4) {
        magic_bits = ((0x7efefefeL << 16) << 16) | 0xfefefeffL;
        himagic = ((himagic << 16) << 16) | himagic;
        lomagic = ((lomagic << 16) << 16) | lomagic;
    }
    if (sizeof (longword) > 8)
        abort ();

    for (;;) {
        longword = *longword_ptr++;

        if (((longword - lomagic) & himagic) != 0) {

            const char *cp = (const char *) (longword_ptr - 1);

            if (cp[0] == 0)
                return cp - str;
            if (cp[1] == 0)
                return cp - str + 1;
            if (cp[2] == 0)
                return cp - str + 2;
            if (cp[3] == 0)
                return cp - str + 3;
            if (sizeof (longword) > 4) {
                if (cp[4] == 0)
                    return cp - str + 4;
                if (cp[5] == 0)
                    return cp - str + 5;
                if (cp[6] == 0)
                    return cp - str + 6;
                if (cp[7] == 0)
                    return cp - str + 7;
            }
        }
    }
}

size_t
naive_strlen(const char *str) {
    const char *s;
    for (s = str; *s; ++s);
    return (s - str);
}

char *
make_buffer(size_t size) {
    char *mem = (char *)malloc(size + 1);
    for (int i = 0; i < size; i++) {
        mem[i] = (rand() % 255) + 1;
    }
    mem[size] = 0;
    return mem;
}

void
run_test() {
    char *buf = make_buffer(BUF_SIZE);
    size_t tot = 0;
    for (int i = 0; i < N_CALLS; i++) {
        //tot += naive_strlen(buf);
        tot += optimized_strlen(buf);
    }
    free(buf);
    printf("%zu\n", tot);
}

int
main(int argc, char *argv[]) {
    PRINT_RUN(run_test);
    return 0;
}
