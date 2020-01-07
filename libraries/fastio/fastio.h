// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#ifndef FASTIO_H
#define FASTIO_H

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define putchar_unlocked putchar
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

// Fast IO routines accessing stdin and stdout. The purpose of them is
// to reduce IO overhead in competitive programming. Therefore, no
// error checking anywhere.

// On Linux and OS X, this is a pointer to stdin's next character. On
// Windows, standard getchar() is used and the pointer
// is NULL.
extern char *
FAST_IO_STDIN;

void
fast_io_init();

// Reading
inline char
fast_io_read_char() {
#ifdef _WIN32
    return getchar();
#else
    return *FAST_IO_STDIN++;
#endif
}

inline unsigned int
fast_io_read_unsigned_int() {
    int val = 0;
#ifdef _WIN32
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

inline int
fast_io_read_int() {
    int val = 0;
    int sgn = 1;

#ifdef _WIN32
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

// Writing
inline void
fast_io_write_char(char ch) {
    putchar_unlocked(ch);
}

inline void
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
    }
    while (rev != 0) {
        putchar_unlocked(rev % 10 + '0');
        rev /= 10;
    }
    while (count--) {
        putchar_unlocked('0');
    }
}

#endif
