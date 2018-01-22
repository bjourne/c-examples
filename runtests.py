from glob import glob
from os import X_OK, access, system, walk
from os.path import join

def all_unit_tests():
    for root, _, files in walk('build/tests'):
        for name in files:
            path = join(root, name)
            if access(path, X_OK):
                yield path

for test in all_unit_tests():
    system(test)
