# C examples
This project contains C code I have written. They say that only the
one who reinvents the wheel truly understand how it works and in this
project I spend a lot of time reinventing wheels.

The project consists of the following parts.

## Libraries
Currently six libraries I have developed while writing garbage
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
datatypes. Instead, I hope that the code is self-explanatory.

### `libraries/isect`

A collection of ray/triangle intersection algorithms. These are used by my raytracer.

### `libraries/linalg`

Trival linear algebra (3D) library. Contains things like matrix
multiplication and stuff.

### `libraries/quickfit`

A memory allocator based on the Quick Fit algorithm. It's used by my
garbage collectors.

### `libraries/file3d`

A library for loading 3d meshes.

## Tests
Test suites for the various libraries.

## Programs
Demo programs of all kinds.

### Raytracer
A simple raytracer I built for a project.

## Install instructions
The project is built using
the [WAF](https://github.com/waf-project/waf) tool. It requires you to
have Python installed.

First configure it:

    $ ./waf configure

The `--isect` and `--shading` options can be passed to control the
compilation of the toy raytracer. Then build:

    $ ./waf build

All build artefacts are put in the `./build` directory.
