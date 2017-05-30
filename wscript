from os.path import splitext

def options(ctx):
    ctx.load('compiler_c compiler_cxx')

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
    extra_flags = debug_flags
    ctx.env.append_unique('CFLAGS',
                          base_flags + extra_flags)
    ctx.env.append_unique('CXXFLAGS', base_flags + extra_flags)
    ctx.env.append_value('INCLUDES', ['libraries'])
    if ctx.env.DEST_OS != 'win32':
        ctx.check(lib = 'pcre')
    ctx.check(lib = 'm')

def build_library(ctx, path, target):
    objs = ctx.path.ant_glob('%s/*.c' % path)
    ctx.objects(source = objs, target = target)

def build_tests(ctx, path, use):
    tests = ctx.path.ant_glob('%s/*.c' % path)
    for test in tests:
        from_path = test.path_from(ctx.path)
        target = splitext(from_path)[0]
        ctx.program(source = [test], use = use, target = target)

def build_program(ctx, filename, use):
    source = 'programs/%s' % filename
    target = 'programs/%s' % splitext(filename)[0]
    ctx.program(source = source, target = target, use = use)

def build(ctx):
    build_library(ctx, 'libraries/datatypes', 'DT_OBJS')
    build_library(ctx, 'libraries/quickfit', 'QF_OBJS')
    build_library(ctx, 'libraries/collectors', 'GC_OBJS')
    build_library(ctx, 'libraries/linalg', 'LINALG_OBJS')

    build_tests(ctx, 'tests/datatypes', ['DT_OBJS'])
    build_tests(ctx, 'tests/quickfit', ['DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'tests/collectors', ['GC_OBJS', 'DT_OBJS', 'QF_OBJS'])
    build_tests(ctx, 'tests/linalg', ['LINALG_OBJS', 'DT_OBJS'])

    build_program(ctx, 'memperf.c', ['DT_OBJS'])
    build_program(ctx, 'multimap.cpp', ['DT_OBJS'])
    if ctx.env.DEST_OS != 'win32':
        build_program(ctx, 'capstack.c', ['DT_OBJS', 'GC_OBJS', 'QF_OBJS'])
        build_program(ctx, 'sigsegv.c', [])
        build_program(ctx, 'pcre.c', ['PCRE'])
    build_program(ctx, 'raytrace.c', ['DT_OBJS', 'M'])
