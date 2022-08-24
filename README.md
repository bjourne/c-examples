# C examples

This repository contains C code that I have written. A lot of it is my
own versions of standard datatypes and algorithms such as my own
hashset datatype. It was written for fun because I love C coding.

As some smart person once said; only the one who reinvents the wheel
truly understand how it works. So there's that.

This repository consists of the following directories:

## Libraries

Ten small libraries I have written while learning about algorithms,
data structures and APIs. They are located in the `libraries`
directory. They are designed to be self-contained and thus easy to
borrow by copying the files.

No guarantee that the libraries are, or ever will be, complete. They
were written because I wanted to.

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

### `libraries/fastio`

Fast IO routines accessing `stdin` and `stdout`. They are useful to
minimize IO overhead in competitive programming challenges.

### `libraries/isect`

A collection of ray/triangle intersection algorithms. These are used
by my raytracer.

### `libraries/linalg`

Trival single-precision linear algebra library. Contains things like
matrix multiplication and stuff.

### `libraries/quickfit`

A memory allocator based on the Quick Fit algorithm. It's used by my
garbage collectors.

### `libraries/file3d`

A library for loading 3d meshes.

### `libraries/diophantine`

A library for solving linear Diophantine equations.

### `libraries/ieee754`

A library for explicit handling of ieee754 floating point numbers.

### `libraries/tensors`

A library for dealing with N-dimensional arrays (tensors).

## Tests
Test suites for the various libraries.

## Programs
Demo programs of all kinds:

* `opencl.c`: Lists installed OpenCL platforms.

## Compilation

The project is built using the build tool
[Waf](https://github.com/waf-project/waf) like this:

    ./waf configure build

Windows, Linux and OS X is supported.
