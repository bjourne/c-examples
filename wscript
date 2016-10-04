from os import path

def options(ctx):
    ctx.load('compiler_c')

def configure(ctx):
    ctx.load('compiler_c')
    base_flags = ['-Wall', '-Werror', '-fPIC']
    debug_flags = ['-O2', '-g']
    ctx.env.append_unique('CFLAGS', base_flags + debug_flags)

def build(ctx):
    datatypes = ['bstree.c', 'common.c', 'hashset.c', 'heap.c', 'vector.c']
    datatypes = [path.join('datatypes', f) for f in datatypes]
    ctx.objects(source = datatypes, target = 'DT_OBJS')
    for prog in ['test-bstree', 'test-hashset', 'test-heap', 'test-vector']:
        main = path.join('datatypes', 'tests', prog + '.c')
        ctx.program(source = [main], use = 'DT_OBJS', target = prog)

    ctx.objects(source = ['quickfit/quickfit.c'], includes = ['.'],
                target = 'QF_OBJS')
    ctx.program(source = ['quickfit/test-quickfit.c'],
                target = 'test-quickfit',
                includes = ['.'], use = ['QF_OBJS', 'DT_OBJS'])
