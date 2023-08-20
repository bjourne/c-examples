# Copyright (C) 2019-2020, 2022-2023 Björn A. Lindqvist <bjourne@gmail.com>
from os.path import splitext
from pathlib import Path
from subprocess import PIPE, Popen

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
            '-Wall', '-Werror',
            # Do I want -fPIC? I don't know.
            '-std=gnu11',
            '-Wsign-compare',
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
    extra_flags = speed_flags
    ctx.env.append_unique('CFLAGS', base_c_flags + extra_flags)
    ctx.env.append_unique('CXXFLAGS', base_cxx_flags + extra_flags)

    ctx.env.append_value('INCLUDES', ['libraries'])
    dest_os = ctx.env.DEST_OS
    if dest_os == 'linux':
        ctx.check(lib = 'X11', mandatory = False)
        ctx.check(lib = 'GL', mandatory = False)
    if dest_os != 'win32':
        ctx.find_program('aoc', var='AOC', mandatory = False)
        ctx.find_program('aocl', var='AOCLEXE', mandatory = False)
        if ctx.env['AOCLEXE']:
            ctx.check_cfg(path = 'aocl', args = 'compile-config',
                          package = '', uselib_store = 'AOCL', mandatory = False)
            ctx.check_cfg(path = 'aocl', args = 'linkflags',
                          package = '', uselib_store = 'AOCL', mandatory = False)

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
        ctx.check(lib = 'OpenCL', mandatory = True, use = ['AOCL'])
        ctx.check(header_name = 'CL/cl.h', use = ['AOCL'])

def noinst_program(ctx, source, target, use):
    ctx.program(source = source, target = target,
                use = sorted(use), install_path = None)

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

def build_library(ctx, libname, target, uses, defines):
    path = 'libraries/%s' % libname
    defs_file = '%s/%s.def' % (path, libname)
    objs = ctx.path.ant_glob('%s/*.c' % path)

    # https://gitlab.com/ita1024/waf/-/issues/2412
    uses = sorted(uses)
    ctx(features = 'c',
        source = objs,
        use = uses,
        target = target,
        defines = defines)
    ctx(features = 'c cstlib',
        target = libname,
        use = [target] + uses,
        defs = defs_file,
        install_path = '${LIBDIR}')

    # Installation of header files
    ctx.install_files('${PREFIX}/include/' + libname,
                      ctx.path.ant_glob('%s/*.h' % path))

def build_aoc(ctx, src, deps):
    tgt = src.with_suffix('.aocx')
    ctx(rule = '${AOC} -march=emulator ${SRC[0]} -o ${TGT} -report',
        source = [str(s) for s in [src] + deps],
        target = str(tgt))

def cc_native_family(cc):
    cmd = [cc, '-march=native', '-E', '-v', '-']
    proc = Popen(cmd,
                 stdin = PIPE, stdout = PIPE, stderr = PIPE, text = True)
    _, stderr = proc.communicate(input = '')

    cc_name = Path(cc).stem
    for l in stderr.splitlines():
        line_parts = l.split()
        if cc_name == 'gcc':
            for item in line_parts:
                if cc_name == 'gcc':
                    parts = item.split('=')
                    if len(parts) == 2 and parts[0] == '-march':
                        return parts[1]
        elif cc_name == 'clang':
            for k, v in zip(line_parts, line_parts[1:]):
                if k == '-target-cpu':
                    return v
    return 'unknown'

def collect_benchmark_flags(env):
    cc_name = env['CC'][0]
    cflags = env['CFLAGS']
    cflags_str = ' '.join(cflags)
    flags = [('CC', cc_name),
             ('CC_NATIVE_FAMILY', cc_native_family(cc_name)),
             ('CFLAGS', cflags_str)]
    flags = ['BENCHMARK_%s="%s"' % (k, v) for k, v in flags]
    return flags

def build(ctx):
    benchmark_flags = collect_benchmark_flags(ctx.env)
    libs = {
        'benchmark' : ('BENCHMARK_OBJS', {}, benchmark_flags),
        'collectors' : ('GC_OBJS', {'QF_OBJS'}, []),
        'datatypes' : ('DT_OBJS', {}, []),
        'diophantine' : ('DIO_OBJS', {}, []),
        'fastio' : ('FASTIO_OBJS', {}, []),
        'file3d' : ('FILE3D_OBJS',
                    {'PATHS_OBJS', 'DT_OBJS', 'LINALG_OBJS'}, []),
        'files' : ('FILES_OBJS', {}, []),
        'isect' : ('ISECT_OBJS', {}, []),
        'linalg' : ('LINALG_OBJS', {}, []),
        'ieee754' : ('IEEE754_OBJS', {}, []),
        # When not using aocl, AOCL will be empty and -lOpenCL will be
        # found by other means.
        'opencl' : ('OPENCL_OBJS',
                    {'AOCL', 'DT_OBJS', 'FILES_OBJS', 'OPENCL'}, []),
        'paths' : ('PATHS_OBJS', {}, []),
        'quickfit' : ('QF_OBJS', ['DT_OBJS'], []),
        'npy' : ('NPY_OBJS', {'M'}, []),
        'random' : ('RANDOM_OBJS', {}, []),
        'tensors' : ('TENSORS_OBJS', {'PNG', 'RANDOM_OBJS'}, []),
        'threads' : ('THREADS_OBJS', {}, [])
    }
    for lib, (sym, deps, defs) in libs.items():
        build_library(ctx, lib, sym, deps, defs)

    # Building all tests here
    tests = {
        'benchmark' : ['BENCHMARK_OBJS', 'DT_OBJS'],
        'collectors' : ['GC_OBJS', 'DT_OBJS', 'QF_OBJS'],
        'datatypes' : ['BENCHMARK_OBJS', 'DT_OBJS', 'RANDOM_OBJS'],
        'ieee754' : ['IEEE754_OBJS', 'DT_OBJS', 'RANDOM_OBJS'],
        'npy' : ['DT_OBJS', 'NPY_OBJS'],
        'opencl' : {
            'DT_OBJS',
            'GOMP',
            'M',
            'OPENCL',
            'OPENCL_OBJS',
            'PATHS_OBJS',
            'PNG',
            'TENSORS_OBJS'
        },
        'quickfit' : ['DT_OBJS', 'QF_OBJS'],
        'random' : {'DT_OBJS', 'RANDOM_OBJS'}
    }
    for lib, deps in tests.items():
        build_tests(ctx, lib, deps)

    build_tests(ctx, 'linalg', ['LINALG_OBJS', 'DT_OBJS', 'M'])
    build_tests(ctx, 'file3d', [
        'FILE3D_OBJS', 'DT_OBJS', 'LINALG_OBJS', 'M'
    ])
    build_tests(ctx, 'isect',
                ['LINALG_OBJS', 'DT_OBJS', 'M', 'ISECT_OBJS'])
    build_tests(ctx, 'paths',
                ['PATHS_OBJS', 'DT_OBJS'])
    build_tests(ctx, 'fastio', ['FASTIO_OBJS'])
    build_tests(ctx, 'files', ['DT_OBJS', 'FILES_OBJS'])

    build_tests(ctx, 'diophantine', ['DIO_OBJS', 'DT_OBJS', 'M'])
    build_tests(ctx, 'tensors', ['TENSORS_OBJS', 'DT_OBJS', 'PNG', 'M'])

    build_program(ctx, 'memperf.c', ['DT_OBJS'])
    build_program(ctx, 'multimap.cpp', ['DT_OBJS'])
    build_program(ctx, 'smallpt.cpp', ['GOMP'])
    noinst_program(ctx, ['programs/ntimes.c',
                         'programs/ntimes-loops.c'],
                   'programs/ntimes',
                   ['DT_OBJS', 'RANDOM_OBJS', 'THREADS_OBJS', 'PTHREAD'])

    progs = [
        ('cpu.c', {'DT_OBJS'}),
        ('fenwick.c', {'FASTIO_OBJS'}),
        ('npyread.c', ['NPY_OBJS']),
        ('simd.c', []),
        ('prodcon.c', {'DT_OBJS', 'THREADS_OBJS', 'RANDOM_OBJS'}),
        # Old fast strlen
        ('strlen.c', ['DT_OBJS']),
        # New fast strlen
        ('fast-strlen.c', ['DT_OBJS', 'RANDOM_OBJS', 'THREADS_OBJS']),
        ('yahtzee.c', ['DT_OBJS', 'THREADS_OBJS', 'PTHREAD'])
    ]
    for cfile, deps in progs:
        build_program(ctx, cfile, deps)

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

    if ctx.env['AOC']:
        kernels = [
            ('dct8x8.cl', []),
            ('matmul_fpga.cl', ['matmul_fpga_config.h']),
            ('pipes.cl', [])
        ]
        base = Path('programs/opencl')
        for kernel, deps in kernels:
            src = Path('programs/opencl') / kernel
            deps = [base / d for d in deps]
            build_aoc(ctx, src, deps)
