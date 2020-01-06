// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#ifndef FASTIO_H
#define FASTIO_H

// Fast IO routines accessing stdin and stdout. The purpose of them
// is to reduce IO overhead in competitive programming.
void fast_io_init();

// Reading
char fast_io_read_char();
void fast_io_write_char(char ch);
unsigned int fast_io_read_unsigned_int();
int fast_io_read_int();

// Writing
void fast_io_write_char(char ch);
void fast_io_write_long(long n);

#endif
