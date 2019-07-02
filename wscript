from os.path import splitext

def options(ctx):
    ctx.load('compiler_c compiler_cxx')

def configure(ctx):
    ctx.load('compiler_c compiler_cxx')
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
        base_c_flags = ['-Wall', '-Werror', '-fPIC', '-std=gnu11']
        base_cxx_flags = ['-Wall', '-Werror', '-fPIC']
        speed_flags = ['-O3',
                       '-fomit-frame-pointer',
                       '-march=native',
                       '-mtune=native']
        debug_flags = ['-O2', '-g']
    extra_flags = speed_flags
    ctx.env.append_unique('CFLAGS', base_c_flags + extra_flags)
    ctx.env.append_unique('CXXFLAGS', base_cxx_flags + extra_flags)
    ctx.env.append_value('INCLUDES', ['libraries'])
    if ctx.env.DEST_OS == 'linux':
        ctx.check(lib = 'X11')
        ctx.check(lib = 'GL')
    if ctx.env.DEST_OS != 'win32':
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
        ctx.check(lib = 'm')

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
    ctx(features = 'c cshlib',
        target = libname,
        use = [target] + uses,
        defs = defs_file)

    # Installation of header files
    ctx.install_files('${PREFIX}/include/' + libname,
                      ctx.path.ant_glob('%s/*.h' % path))

def build(ctx):
    build_library(ctx, 'datatypes', 'DT_OBJS', [])
    build_library(ctx, 'quickfit', 'QF_OBJS', ['datatypes'])
    build_library(ctx, 'collectors', 'GC_OBJS', ['quickfit'])
    build_library(ctx, 'linalg', 'LINALG_OBJS', ['M'])
    build_library(ctx, 'isect', 'ISECT_OBJS', [])
    build_library(ctx, 'file3d', 'FILE3D_OBJS',
                  ['datatypes', 'linalg'])

    build_tests(ctx, 'datatypes', ['DT_OBJS'])
    build_tests(ctx, 'quickfit', ['DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'collectors', ['GC_OBJS', 'DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'linalg', ['LINALG_OBJS', 'DT_OBJS', 'M'])
    build_tests(ctx, 'file3d', ['FILE3D_OBJS', 'DT_OBJS'])
    build_tests(ctx, 'isect',
                ['LINALG_OBJS', 'DT_OBJS', 'M', 'ISECT_OBJS'])

    build_program(ctx, 'cpu.c', ['DT_OBJS'])
    build_program(ctx, 'memperf.c', ['DT_OBJS'])
    build_program(ctx, 'multimap.cpp', ['DT_OBJS'])
    build_program(ctx, 'simd.c', [])

    if ctx.env.DEST_OS == 'linux':
        build_program(ctx, 'sigsegv.c', [])
        build_program(ctx, 'gl-fbconfigs.c', ['GL', 'X11'])
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
