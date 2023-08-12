# C examples

This repository contains my C code. A lot of it is my own versions of
standard datatypes and algorithms such as my own hashset datatype. It
was written for fun because I love C coding.

As some smart person once said; only the one who reinvents the wheel
truly understand how it works. So there's that.

This repository consists of the following directories:

## Libraries

Located in `libraries` are a bunch of libraries I have written to
learn more about algorithms, data structures, OS APIs, etc. They are
designed to be mostly self-contained and thus easy to borrow by
copying the files. However, over the years I have introduced
dependencies between them. Factoring out common code is just too
pleasurable.

No guarantee that the libraries are -- or ever will be --
complete. They were written because I wanted to.

### `libraries/benchmark`

Minimal library for benchmarking C code.

### `libraries/collectors`

An object model and a bunch of garbage collectors. These were written
to teach myself about garbage collection strategies.

* `copying.[ch]` - Bump pointer allocation and semi-space copying
* `copying-opt.[ch]` - Optimized version of the above
* `ref-counting.[ch]` - Plain reference counting
* `ref-counting-cycles.[ch]` - Reference counting with cycle detection
* `mark-sweep.[ch]` - Mark & Sweep gc

### `libraries/datatypes`

Standard datatypes for C programming like `vector` and
`hashset`. There is not a lot of documentation for these
datatypes because the code should be self-explanatory.

### `libraries/diophantine`

A minimal library for solving linear Diophantine equations.

### `libraries/fastio`

Fast IO routines accessing `stdin` and `stdout`. They are useful to
minimize IO overhead in competitive programming challenges.

### `libraries/file3d`

A library for loading 3d meshes.

### `libraries/ieee754`

A library for explicit handling of ieee754 floating point numbers.

### `libraries/isect`

A collection of ray/triangle intersection algorithms. These are used
by my raytracer.

### `libraries/linalg`

Trival single-precision linear algebra library. Contains things like
matrix multiplication and stuff.

### `libraries/opencl`

A library containing OpenCL utility functions. It can be tricky to
compile for Intel FPGA because libraries and headers are not in
standard locations. You can try:

    CFLAGS=$(aocl compile-config) \
        CXXFLAGS=$(aocl compile-config) \
        LDFLAGS=$(aocl ldflags) \
        ./waf-2.0.25 build

Of course, this assumes that `aocl` and related programs are on the
`PATH`. To run the built programs the OpenCL library must be linkable:

    LD_LIBRARY_PATH=/path/to/opencl ./build/programs/opencl/prog

### `libraries/random`

Random number generation using [www.pcg-random.org](www.pcg-random.org).

### `libraries/paths`

A library for filesystem path handling.

### `libraries/quickfit`

A memory allocator based on the Quick Fit algorithm. It's used by my
garbage collectors.

### `libraries/tensors`

A library for dealing with N-dimensional arrays (tensors).

## Tests
Test suites for the various libraries.

## Programs
Demo programs of all kinds:

* `programs/npyread.c`: Reading .npy files.
* `programs/ntimes.c`: See [{n} times faster than C - part one](https://owen.cafe/posts/six-times-faster-than-c/).
* `programs/opencl/list.c`: List installed OpenCL platforms.

## Compilation

The project is built using the build tool
[Waf](https://github.com/waf-project/waf) like this:

    ./waf configure build

Windows, Linux and OS X is supported.

## Design Philosophy

Structs are not private. I have good reasons for making them public,
but I forgot what they are now.
