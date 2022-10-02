# Copyright (C) 2019-2020, 2022 Björn A. Lindqvist <bjourne@gmail.com>
from os.path import splitext

def options(ctx):
    ctx.load('compiler_c compiler_cxx')

def configure(ctx):
    ctx.load('compiler_c compiler_cxx')
    ctx.define('_GNU_SOURCE', 1)
    if ctx.env.CC_NAME == 'msvc':
        base_c_flags = [
            '/WX', '/W3', '/O2', '/EHsc',
            # Without these flags, msvc generates a billion bullshit
            # warnings.
            '/D_CRT_SECURE_NO_WARNINGS',
            '/D_CRT_NONSTDC_NO_DEPRECATE'
        ]
        base_cxx_flags = base_c_flags
        debug_flags = ['/Zi', '/FS']
        speed_flags = []
    else:
        base_c_flags = [
            '-Wall', '-Werror', '-fPIC', '-std=gnu11',
            # Since we are now using SIMD intrinsics
            '-march=native', '-mtune=native',
            '-fopenmp'
        ]
        base_cxx_flags = [
            '-Wall', '-Werror', '-fPIC', '-fopenmp',
            '-march=native', '-mtune=native'
        ]
        speed_flags = ['-O3', '-fomit-frame-pointer']
        debug_flags = ['-O2', '-g']
    extra_flags = debug_flags
    ctx.env.append_unique('CFLAGS', base_c_flags + extra_flags)
    ctx.env.append_unique('CXXFLAGS', base_cxx_flags + extra_flags)
    ctx.env.append_value('INCLUDES', ['libraries'])
    dest_os = ctx.env.DEST_OS
    if dest_os == 'linux':
        ctx.check(lib = 'X11', mandatory = False)
        ctx.check(lib = 'GL', mandatory = False)
    if dest_os != 'win32':
        llvm_libs = ['core', 'executionengine', 'mcjit', 'native']
        args = [
            '--cflags',
            '--ldflags',
            '--libs',
            ' '.join(llvm_libs),
            '--system-libs'
        ]
        ctx.check_cfg(path = 'llvm-config-3.9',
                      args = ' '.join(args),
                      package = '',
                      uselib_store = 'LLVM',
                      mandatory = False,
                      msg = 'Checking for \'llvm-3.9\'')
        ctx.check_cfg(package = 'libpcre',
                      args = ['libpcre >= 8.33', '--cflags', '--libs'],
                      uselib_store = 'PCRE',
                      mandatory = False)
        ctx.check_cfg(package = 'libpng',
                      args = ['--libs', '--cflags'],
                      uselib_store = 'PNG',
                      mandatory = False)
        ctx.check(lib = 'gomp', mandatory = True, uselib_store = 'GOMP')
        ctx.check(lib = 'm', mandatory = False)
        ctx.check(lib = 'pthread', mandatory = False)
        ctx.check(lib = 'OpenCL', mandatory = True)


def noinst_program(ctx, source, target, use):
    ctx.program(source = source, target = target,
                use = use, install_path = None)

def build_objects(ctx, path, target):
    path = 'libraries/%s' % path
    objs = ctx.path.ant_glob('%s/*.c' % path)
    ctx.objects(source = objs, target = target)

def build_tests(ctx, path, use):
    path = 'tests/%s' % path
    tests = ctx.path.ant_glob('%s/*.c' % path)
    for test in tests:
        from_path = test.path_from(ctx.path)
        target = splitext(from_path)[0]
        noinst_program(ctx, [test], target, use)

def build_program(ctx, fname, use):
    source = 'programs/%s' % fname
    target = 'programs/%s' % splitext(fname)[0]
    noinst_program(ctx, source, target, use)

def build_library(ctx, libname, target, uses):
    path = 'libraries/%s' % libname
    defs_file = '%s/%s.def' % (path, libname)
    objs = ctx.path.ant_glob('%s/*.c' % path)

    ctx(features = 'c', source = objs, target = target)
    ctx(features = 'c cstlib',
        target = libname,
        use = [target] + uses,
        defs = defs_file,
        install_path = '${LIBDIR}')

    # Installation of header files
    ctx.install_files('${PREFIX}/include/' + libname,
                      ctx.path.ant_glob('%s/*.h' % path))

def build(ctx):
    build_library(ctx, 'datatypes', 'DT_OBJS', [])
    build_library(ctx, 'quickfit', 'QF_OBJS', ['DT_OBJS'])
    build_library(ctx, 'collectors', 'GC_OBJS', ['QF_OBJS'])
    build_library(ctx, 'linalg', 'LINALG_OBJS', ['DT_OBJS', 'M'])
    build_library(ctx, 'fastio', 'FASTIO_OBJS', [])
    build_library(ctx, 'isect', 'ISECT_OBJS', [])
    build_library(ctx, 'file3d', 'FILE3D_OBJS',
                  ['DT_OBJS', 'LINALG_OBJS'])
    build_library(ctx, 'paths', 'PATHS_OBJS', [])

    build_library(ctx, 'threads', 'THREADS_OBJS', [])
    build_library(ctx, 'diophantine', 'DIO_OBJS', [])
    build_library(ctx, 'ieee754', 'IEEE754_OBJS', [])
    build_library(ctx, 'tensors', 'TENSORS_OBJS', ['PNG'])
    build_library(ctx, 'opencl', 'OPENCL_OBJS', ['OPENCL', 'DT_OBJS'])

    build_tests(ctx, 'datatypes', ['DT_OBJS'])
    build_tests(ctx, 'quickfit', ['DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'collectors', ['GC_OBJS', 'DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'linalg', ['LINALG_OBJS', 'DT_OBJS', 'M'])
    build_tests(ctx, 'file3d', [
        'FILE3D_OBJS', 'DT_OBJS', 'LINALG_OBJS', 'M'
    ])
    build_tests(ctx, 'isect',
                ['LINALG_OBJS', 'DT_OBJS', 'M', 'ISECT_OBJS'])
    build_tests(ctx, 'paths',
                ['PATHS_OBJS', 'DT_OBJS'])
    build_tests(ctx, 'fastio', ['FASTIO_OBJS'])

    build_tests(ctx, 'diophantine', ['DIO_OBJS', 'DT_OBJS', 'M'])
    build_tests(ctx, 'ieee754', ['IEEE754_OBJS', 'DT_OBJS'])
    build_tests(ctx, 'tensors', ['TENSORS_OBJS', 'DT_OBJS', 'PNG', 'M'])
    build_tests(ctx, 'opencl', [
        'OPENCL_OBJS',
        'PATHS_OBJS',
        'TENSORS_OBJS',
        'OPENCL', 'PNG', 'M', 'GOMP',
        'DT_OBJS'
    ])

    build_program(ctx, 'cpu.c', ['DT_OBJS'])
    build_program(ctx, 'memperf.c', ['DT_OBJS'])
    build_program(ctx, 'multimap.cpp', ['DT_OBJS'])
    build_program(ctx, 'smallpt.cpp', ['GOMP'])
    build_program(ctx, 'simd.c', [])
    build_program(ctx, 'strlen.c', ['DT_OBJS'])
    build_program(ctx, 'fenwick.c', ['FASTIO_OBJS'])
    build_program(ctx, 'yahtzee.c', ['DT_OBJS', 'THREADS_OBJS', 'PTHREAD'])

    # Conditional targets
    if ctx.env.DEST_OS == 'linux':
        build_program(ctx, 'sigsegv.c', [])
        build_program(ctx, 'opencl/list.c', [
            'OPENCL', 'OPENCL_OBJS', 'PATHS_OBJS'
        ])
        build_program(ctx, 'opencl/dct.c', [
            'OPENCL', 'OPENCL_OBJS', 'PATHS_OBJS',
            'TENSORS_OBJS', 'PNG', 'M', 'GOMP',
            'DT_OBJS'
        ])
        build_program(ctx, 'opencl/fpga.c', [
            'OPENCL', 'OPENCL_OBJS', 'PATHS_OBJS',
            'TENSORS_OBJS', 'PNG', 'M', 'DT_OBJS'
        ])
    if ctx.env.DEST_OS != 'win32':
        build_program(ctx, 'capstack.c',
                      ['DT_OBJS', 'GC_OBJS', 'QF_OBJS'])
    else:
        build_program(ctx, 'winthreads.c', [])

    if ctx.env['LIB_PCRE']:
        build_program(ctx, 'pcre.c', ['PCRE'])
    if ctx.env['LIB_LLVM']:
        build_program(ctx, 'llvm-wbc.c', ['LLVM'])
        build_program(ctx, 'llvm-rbc.c', ['LLVM'])
    if ctx.env['LIB_GL'] and ctx.env['LIB_X11']:
        build_program(ctx, 'gl-fbconfigs.c', ['GL', 'X11'])
