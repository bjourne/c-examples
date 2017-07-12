from os.path import splitext

def options(ctx):
    ctx.load('compiler_c compiler_cxx')
    ctx.add_option(
        '--isect',
        type = 'string',
        default = 'ISECT_MT',
        help = 'Preferred intersection algorithm.'
        )
    ctx.add_option(
        '--shading',
        type = 'string',
        default = 'FANCY_SHADING',
        help = 'Preferred shading style.'
        )

def configure(ctx):
    ctx.load('compiler_c compiler_cxx')
    if ctx.env.DEST_OS == 'win32':
        base_flags = ['/WX', '/W3', '/O2', '/EHsc']
        debug_flags = ['/Zi', '/FS']
        speed_flags = []
    else:
        base_flags = ['-Wall', '-Werror', '-fPIC']
        speed_flags = ['-O3',
                       '-fomit-frame-pointer',
                       '-march=native',
                       '-mtune=native']
        debug_flags = ['-O2', '-g']
    extra_flags = speed_flags
    ctx.env.append_unique('CFLAGS',
                          base_flags + extra_flags)
    ctx.env.append_unique('CXXFLAGS', base_flags + extra_flags)
    ctx.env.append_value('INCLUDES', ['libraries'])
    if ctx.env.DEST_OS != 'win32':
        ctx.check(lib = 'pcre')
        llvm_libs = ['core', 'executionengine', 'mcjit', 'native']
        args = [
            '--cxxflags',
            '--ldflags',
            '--libs',
            ' '.join(llvm_libs),
            '--system-libs'
        ]
        ctx.check_cfg(path = 'llvm-config-3.8',
                      args = ' '.join(args),
                      package = '',
                      uselib_store = 'LLVM')
    ctx.check(lib = 'm')
    ctx.define('ISECT_METHOD', ctx.options.isect, quote = False)
    ctx.define('SHADING_STYLE', ctx.options.shading, quote = False)

def build_library(ctx, path, target):
    path = 'libraries/%s' % path
    objs = ctx.path.ant_glob('%s/*.c' % path)
    ctx.objects(source = objs, target = target)

def build_tests(ctx, path, use):
    path = 'tests/%s' % path
    tests = ctx.path.ant_glob('%s/*.c' % path)
    for test in tests:
        from_path = test.path_from(ctx.path)
        target = splitext(from_path)[0]
        ctx.program(source = [test], use = use, target = target)

def build_program(ctx, fname, use):
    source = 'programs/%s' % fname
    target = 'programs/%s' % splitext(fname)[0]
    ctx.program(source = source, target = target, use = use)

def build(ctx):
    build_library(ctx, 'datatypes', 'DT_OBJS')
    build_library(ctx, 'quickfit', 'QF_OBJS')
    build_library(ctx, 'collectors', 'GC_OBJS')
    build_library(ctx, 'linalg', 'LINALG_OBJS')
    build_library(ctx, 'isect', 'ISECT_OBJS')
    build_library(ctx, 'file3d', 'FILE3D_OBJS')

    build_tests(ctx, 'datatypes', ['DT_OBJS'])
    build_tests(ctx, 'quickfit', ['DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'collectors', ['GC_OBJS', 'DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'linalg', ['LINALG_OBJS', 'DT_OBJS', 'M'])
    build_tests(ctx, 'file3d', ['FILE3D_OBJS', 'DT_OBJS'])

    build_program(ctx, 'memperf.c', ['DT_OBJS'])
    build_program(ctx, 'multimap.cpp', ['DT_OBJS'])
    if ctx.env.DEST_OS != 'win32':
        build_program(ctx, 'capstack.c',
                      ['DT_OBJS', 'GC_OBJS', 'QF_OBJS'])
        build_program(ctx, 'sigsegv.c', [])
        build_program(ctx, 'pcre.c', ['PCRE'])
        build_program(ctx, 'llvm-wbc.cpp', ['LLVM'])
        build_program(ctx, 'llvm-rbc.cpp', ['LLVM'])
    # Raytracer
    source = ctx.path.ant_glob('programs/raytrace/*.c')
    ctx.program(source = source,
                target = 'rt',
                use = [
                    'DT_OBJS',
                    'FILE3D_OBJS',
                    'M',
                    'LINALG_OBJS',
                    'ISECT_OBJS'])

    build_tests(ctx, 'isect',
                ['LINALG_OBJS', 'DT_OBJS', 'M', 'ISECT_OBJS'])
