# Copyright (C) 2019-2020, 2022-2024 Björn A. Lindqvist <bjourne@gmail.com>
from os.path import splitext
from pathlib import Path
from subprocess import PIPE, Popen

def options(ctx):
    for tool in {'compiler_c', 'compiler_cxx', 'pkgconfig'}:
        ctx.load_tool(tool)

def configure(ctx):
    for tool in {'compiler_c', 'compiler_cxx', 'pkgconfig'}:
        ctx.load_tool(tool)
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
        # This stuff interferes with oneAPI
        # ctx.find_program('aoc', var='AOC', mandatory = False)
        # ctx.find_program('aocl', var='AOCLEXE', mandatory = False)
        # if ctx.env['AOCLEXE']:
        #     ctx.check_cfg(path = 'aocl', args = ['compile-config'],
        #                   package = '', uselib_store = 'AOCL', mandatory = False)
        #     ctx.check_cfg(path = 'aocl', args = ['linkflags'],
        #                   package = '', uselib_store = 'AOCL', mandatory = False)
        llvm_libs = ['core', 'executionengine', 'mcjit', 'native']
        llvm_args = [
            '--cflags',
            '--ldflags',
            '--libs'
        ] + llvm_libs + ['--system-libs']
        ctx.check_cfg(path = 'llvm-config',
                      args = llvm_args,
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
        ctx.check(header_name = 'CL/cl.h')
        ctx.check(lib = 'mpi', mandatory = False)

    # We grab compilation and link flags for CUDA and oneAPI by
    # locating respective toolkit's compiler.
    ret = ctx.find_program("icpx", var="ICPX", mandatory = False)
    if ret:
        prefix = Path(ctx.env["ICPX"][0]).parent.parent
        incl_base = prefix / 'include'
        incl_sycl = incl_base / 'sycl'
        includes = [str(incl) for incl in [incl_base, incl_sycl]]
        ctx.check(
            header_name = "CL/sycl.hpp",
            includes = includes,
            lib = 'sycl',
            cxxflags = ['-Wno-attributes', '-Wno-deprecated-declarations'],
            uselib_store = 'SYCL'
        )

    ret = ctx.find_program("nvcc", var="NVCC", mandatory = False)
    if ret:
        prefix = Path(ctx.env["NVCC"][0]).parent.parent
        includes = [str(prefix / "include")]

        libdirs = ["lib64", "lib64/stubs", "lib", "lib/stubs"]
        libpath = [prefix / x for x in libdirs]
        libpath = [str(p) for p in libpath if p.exists()]
        ctx.check(header_name="cuda.h",
                  includes = includes,
                  libpath = libpath,
                  lib = "cudart",
                  uselib_store = 'CUDA')


def noinst_program(ctx, source, target, use, features):
    assert type(source) == list
    source = [str(s) for s in source]
    target = str(target)
    ctx(features = features,
        source = source, target = target,
        use = sorted(use), install_path = None)

def build_tests(ctx, lib, deps):
    path = Path('tests') / lib
    tests = path.glob('*.c')
    for test in tests:
        target = path / test.stem
        noinst_program(ctx, [test], target, deps,
                       ["c", "cprogram"])

def build_program(ctx, source, use):
    ext = source[0].suffix
    progdir = Path('programs')

    # Compute target
    target = progdir / source[0].with_suffix('')

    cprog = ["c", "cprogram"]
    cxxprog = ["cxx", "cxxprogram"]
    features = cprog if ext == ".c" else cxxprog
    source = [progdir / s for s in source] # 'programs/%s' % fname
    noinst_program(ctx, source, target, use, features)

def build_library(ctx, libname, target, uses, defines):
    path = 'libraries/%s' % libname
    defs_file = '%s/%s.def' % (path, libname)
    objs = ctx.path.ant_glob('%s/*.c' % path)

    # https://gitlab.com/ita1024/waf/-/issues/2412
    uses = sorted(uses)
    ctx(features = ['c'],
        source = objs,
        use = uses,
        target = target,
        defines = defines)
    ctx(features = ['c', 'cstlib'],
        target = libname,
        use = [target] + uses,
        defs = defs_file,
        install_path = '${LIBDIR}')

    # Installation of header files
    ctx.install_files('${PREFIX}/include/' + libname,
                      ctx.path.ant_glob('%s/*.h' % path))

def build_aoc(ctx, src, deps):
    aocx = str(src.with_suffix('.aocx'))
    dir = aocx.split('.')[0]

    aoc_cmd = ' '.join([
        "${AOC}",
        "-march=emulator",
        # Include the project's root directory
        "-I..",
        "${SRC[0]}",
        "-o",
        "${TGT[0]}",
        "-report"])

    # Need to remove the log directory otherwise aoc gets angry.
    rules = ['rm -rf ${TGT[1]}', aoc_cmd]
    ctx(rule = ' && '.join(rules),
        source = [str(s) for s in [src] + deps],
        target = [aocx, dir])

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
        'diophantine' : ('DIO_OBJS', {'RANDOM_OBJS'}, []),
        'fastio' : ('FASTIO_OBJS', {}, []),
        'file3d' : ('FILE3D_OBJS',
                    {'PATHS_OBJS', 'DT_OBJS', 'LINALG_OBJS'}, []),
        'files' : ('FILES_OBJS', {}, []),
        'isect' : ('ISECT_OBJS', {}, []),
        'linalg' : ('LINALG_OBJS', {}, []),
        'ieee754' : ('IEEE754_OBJS', {}, []),
        # When not using aocl, AOCL will be empty and -lOpenCL will be
        # found by other means.
        'opencl' : ('OPENCL_OBJS', {
            'AOCL', 'DT_OBJS', 'FILES_OBJS', 'OPENCL',
            'PATHS_OBJS', 'PRETTY_OBJS'
        }, []),
        'paths' : ('PATHS_OBJS', {}, []),
        'pretty' : ('PRETTY_OBJS', {'M'}, []),
        'quickfit' : ('QF_OBJS', {'DT_OBJS'}, []),
        'npy' : ('NPY_OBJS', {'PRETTY_OBJS', 'M'}, []),
        'random' : ('RANDOM_OBJS', {}, []),
        'tensors' : ('TENSORS_OBJS', {'PNG', 'PRETTY_OBJS', 'RANDOM_OBJS'}, []),
        'threads' : ('THREADS_OBJS', {}, [])
    }

    # Building all library tests here
    tests = {
        'benchmark' : ['BENCHMARK_OBJS', 'DT_OBJS'],
        'collectors' : ['GC_OBJS', 'DT_OBJS', 'QF_OBJS'],
        'datatypes' : ['BENCHMARK_OBJS', 'DT_OBJS', 'RANDOM_OBJS'],
        'diophantine' : {'DIO_OBJS', 'DT_OBJS', 'M'},
        'fastio' : {'FASTIO_OBJS'},
        'file3d' : {'FILE3D_OBJS', 'DT_OBJS', 'LINALG_OBJS', 'M'},
        'files' : {'DT_OBJS', 'FILES_OBJS'},
        'ieee754' : ['IEEE754_OBJS', 'DT_OBJS', 'RANDOM_OBJS'],
        'isect' : {'LINALG_OBJS', 'DT_OBJS', 'M', 'ISECT_OBJS'},
        'linalg' : {'LINALG_OBJS', 'DT_OBJS', 'M', 'RANDOM_OBJS'},
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
        'paths' : {'PATHS_OBJS', 'DT_OBJS'},
        'pretty' : {'DT_OBJS', 'PRETTY_OBJS'},
        'quickfit' : ['DT_OBJS', 'QF_OBJS'],
        'random' : {'DT_OBJS', 'RANDOM_OBJS'},
        'tensors' : {'TENSORS_OBJS', 'DT_OBJS', 'PNG', 'M'},
        'threads' : {'DT_OBJS', 'RANDOM_OBJS', 'THREADS_OBJS'}
    }
    if ctx.env['LIB_CUDA']:
        libs['cud'] = ('CUD_OBJS', {"CUDA", "PRETTY_OBJS"}, [])
        tests['cud'] = {'CUD_OBJS', 'DT_OBJS', 'CUDA'}

    for lib, (sym, deps, defs) in libs.items():
        build_library(ctx, lib, sym, deps, defs)
    for lib, deps in tests.items():
        build_tests(ctx, lib, deps)

    progs = [
        (['cpu.c'], {'DT_OBJS'}),
        (['fenwick.c'], {'FASTIO_OBJS'}),
        (['mandelbrot.c'], {'DT_OBJS', 'LINALG_OBJS', 'M', 'TENSORS_OBJS'}),
        (['memperf.c'], ['DT_OBJS']),
        (['multimap.cpp'], ['DT_OBJS']),
        (['ntimes.c', 'ntimes-loops.c'],
         ['PTHREAD', 'DT_OBJS', 'RANDOM_OBJS', 'THREADS_OBJS']),
        (['smallpt.cpp'], ['GOMP']),
        (['npyread.c'], ['NPY_OBJS']),
        (['simd.c'], {}),
        (['prodcon.c'], {'DT_OBJS', 'THREADS_OBJS', 'RANDOM_OBJS'}),
        # Old fast strlen
        (['strlen.c'], {'DT_OBJS'}),
        # New fast strlen
        (['fast-strlen.c'], ['DT_OBJS', 'RANDOM_OBJS', 'THREADS_OBJS']),
        (['yahtzee.c'], ['DT_OBJS', 'THREADS_OBJS', 'PTHREAD']),
    ]


    linux_progs = [
        (['opencl/comm.c'], {'OPENCL', 'OPENCL_OBJS', 'RANDOM_OBJS'}),
        (['opencl/dct.c'], [
            'OPENCL', 'OPENCL_OBJS', 'PATHS_OBJS',
            'TENSORS_OBJS', 'PNG', 'M', 'GOMP',
            'DT_OBJS'
        ]),
        (['opencl/fpga.c'], {
            'OPENCL', 'OPENCL_OBJS', 'PATHS_OBJS',
            'TENSORS_OBJS', 'PNG', 'M', 'DT_OBJS'
        }),
        (['opencl/list.c'], {'OPENCL', 'OPENCL_OBJS', 'PATHS_OBJS'}),
        (['sigsegv.c'], {})
    ]



    if ctx.env.DEST_OS == 'linux':
        progs.extend(linux_progs)

    not_win32_progs = [
        (['capstack.c'], {'DT_OBJS', 'GC_OBJS', 'QF_OBJS'})
    ]
    if ctx.env.DEST_OS != 'win32':
        progs.extend(not_win32_progs)

    win32_progs = [
        (['winthreads.c'], [])
    ]
    if ctx.env.DEST_OS == 'win32':
        progs.extend(win32_progs)

    if ctx.env['LIB_PCRE']:
        progs.append((['pcre.c'], {'PCRE'}))

    if ctx.env['LIB_SYCL']:
        progs.append((['sycl-list.cpp'], {'SYCL', 'PRETTY_OBJS'}))

    if ctx.env['LIB_MPI']:
        progs.extend([
            (['openmpi/heat.c'], {
                'DT_OBJS',
                'RANDOM_OBJS',
                'MPI',
                'TENSORS_OBJS',
                'M'
            }),
            (['openmpi/pi.c'], {'DT_OBJS', 'RANDOM_OBJS', 'MPI'})
        ])

    if ctx.env['LIB_LLVM']:
        progs.extend([
            (['llvm-wbc.c'], ['LLVM']),
            (['llvm-rbc.c'], ['LLVM']),
        ])
    if ctx.env['LIB_GL'] and ctx.env['LIB_X11']:
        progs.append((['gl-fbconfigs.c'], {'GL', 'X11'}))

    progs = [([Path(s) for s in source], deps)
             for (source, deps) in progs]
    for files, deps in progs:
        build_program(ctx, files, deps)

    if ctx.env['AOC']:
        kernels = [
            ('dct8x8.cl', []),
            ('matmul_fpga.cl', ['matmul_fpga_config.h']),
            ('pipes.cl', [])
        ]
        base = Path('programs/opencl')
        for kernel, deps in kernels:
            src = base / kernel
            deps = [base / d for d in deps]
            build_aoc(ctx, src, deps)
