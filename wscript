from os.path import splitext

def options(ctx):
    ctx.load('compiler_c')

def configure(ctx):
    ctx.load('compiler_c')
    base_flags = ['-Wall', '-Werror', '-fPIC']
    debug_flags = ['-O2', '-g']
    ctx.env.append_unique('CFLAGS', base_flags + debug_flags)
    ctx.env.append_value('INCLUDES', ['libraries'])
    ctx.check(lib = 'pcre')

def build_library(ctx, path, target):
    objs = ctx.path.ant_glob('%s/*.c' % path)
    ctx.objects(source = objs, target = target)

def build_tests(ctx, path, use):
    tests = ctx.path.ant_glob('%s/*.c' % path)
    for test in tests:
        from_path = test.path_from(ctx.path)
        target = splitext(from_path)[0]
        ctx.program(source = [test], use = use, target = target)

def build(ctx):
    build_library(ctx, 'libraries/datatypes', 'DT_OBJS')
    build_library(ctx, 'libraries/quickfit', 'QF_OBJS')

    build_tests(ctx, 'tests/datatypes', ['DT_OBJS'])
    build_tests(ctx, 'tests/quickfit', ['DT_OBJS', 'QF_OBJS'])

    for prog in ctx.path.ant_glob('programs/*.c'):
        from_path = prog.path_from(ctx.path)
        target = splitext(from_path)[0]
        ctx.program(source = [prog], target = target, use = ['PCRE', 'DT_OBJS'])
