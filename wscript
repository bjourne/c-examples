from os import path

def options(ctx):
    ctx.load('compiler_c')

def configure(ctx):
    ctx.load('compiler_c')
    base_flags = ['-Wall', '-Werror', '-fPIC']
    debug_flags = ['-O2', '-g']
    ctx.env.append_unique('CFLAGS', base_flags + debug_flags)

def build(ctx):
    datatypes = ['common.c', 'hashset.c', 'vector.c']
    datatypes = [path.join('datatypes', f) for f in datatypes]
    ctx.objects(source = datatypes, target = 'OBJS')
    for prog in ['test-hashset', 'test-vector']:
        main = path.join('datatypes', 'tests', prog + '.c')
        ctx.program(source = [main], use = 'OBJS', target = prog)
