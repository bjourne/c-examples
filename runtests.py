from glob import glob
from os import system
from os.path import join

all_tests_glob = join('build', 'tests', '**', '*.exe')

for fname in glob(all_tests_glob)[:6]:
    system(fname)
