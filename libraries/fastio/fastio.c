// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define putchar_unlocked putchar
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

static char *
FAST_IO_PTR = NULL;

void
fast_io_init() {
    #ifndef _MSC_VER
    struct stat sb;
    (void)fstat(STDIN_FILENO, &sb);
    FAST_IO_STDIN = (char*)mmap(0, sb.st_size,
                              PROT_READ, MAP_SHARED | MAP_POPULATE,
                              STDIN_FILENO, 0);
    #endif
}

void
fast_io_write_long(long n) {
    if (n < 0) {
        putchar_unlocked('-');
        n *= -1;
    } else if (n == 0) {
        putchar_unlocked('0');
        return;
    }
    long N = n;
    long rev = N;
    int count = 0;
    rev = N;

    while ((rev % 10) == 0) {
        count++; rev /= 10;
    }
    rev = 0;
    while (N != 0) {
        rev = (rev<<3) + (rev<<1) + N % 10; N /= 10;
    }  //store reverse of N in rev
    while (rev != 0) {
        putchar_unlocked(rev % 10 + '0');
        rev /= 10;
    }
    while (count--) {
        putchar_unlocked('0');
    }
}

void
fast_io_write_char(char ch) {
    putchar_unlocked(ch);
}

char
fast_io_read_char() {
    #ifdef _MSC_VER
    return getchar();
    #else
    return *FAST_IO_STDIN++;
    #endif
}

unsigned int
fast_io_read_unsigned_int() {
    int val = 0;
    #ifdef _MSC_VER
    while (true) {
        char c = getchar();
        if (c < '0') {
            ungetc(c, stdin);
            break;
        }
        val = val * 10 + c - '0';
    }
    #else
    do {
        val = val*10 + *FAST_IO_STDIN++ - '0';
    } while(*FAST_IO_STDIN >= '0');
    #endif
    return val;
}
int
fast_io_read_int() {
    int val = 0;
    int sgn = 1;

    #ifdef _MSC_VER
    char c = getchar();
    if (c == '-') {
        sgn = -1;
    } else {
        ungetc(c, stdin);
    }
    while (true) {
        char c = getchar();
        if (c < '0') {
            ungetc(c, stdin);
            break;
        }
        val = 10 * val + c - '0';
    }
    #else
    if (*FAST_IO_STDIN == '-') {
        sgn = -1;
        FAST_IO_STDIN++;
    }
    do {
        val = val*10 + *FAST_IO_STDIN++ - '0';
    } while(*FAST_IO_STDIN >= '0');
    #endif
    return val * sgn;
}
