# C examples

This repository contains C code that I have written. A lot of it is my
own versions of standard datatypes and algorithms such as my own
hashset datatype. It was written for fun because I love C coding.

As some smart person once said; only the one who reinvents the wheel
truly understand how it works. So there's that.

The repository consists of the following parts.

## Libraries
Currently seven libraries I have developed while writing garbage
collectors and other projects. They are located in the `libraries`
directory.

No guarantee that the libraries are, or ever will be, complete. They
were written because I personally needed them.

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

## Tests
Test suites for the various libraries.

## Programs
Demo programs of all kinds.

## Compilation

The project is built using the build tool
[Waf](https://github.com/waf-project/waf) like this:

    ./waf configure build

Windows, Linux and OS X is supported.
