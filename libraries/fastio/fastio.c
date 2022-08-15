// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#include "fastio.h"

char *
FAST_IO_STDIN = NULL;

void
fast_io_init() {
#ifdef _WIN32
    // Nothing to do on Windows because we use standard getchar().
#else
    int flags = MAP_SHARED;
#ifdef __linux__
    // Prefills the buffer.
    flags |= MAP_POPULATE;
#endif
    struct stat sb;
    (void)fstat(STDIN_FILENO, &sb);
    FAST_IO_STDIN = (char *)mmap(0, sb.st_size,
                                 PROT_READ, flags, STDIN_FILENO, 0);
#endif
}

extern inline char fast_io_read_char();
extern inline unsigned int fast_io_read_unsigned_int();
extern inline int fast_io_read_int();
extern inline void fast_io_write_char(char ch);
extern inline void fast_io_write_long(long n);
