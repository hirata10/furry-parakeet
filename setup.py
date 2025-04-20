# setup script for pyimcom_croutines

import os
from os.path import exists
from setuptools import Extension, setup

# get include directories
dirs = [];
if exists('IncludePaths.txt'):
  with open('IncludePaths.txt') as f:
    lines = f.readlines()
    for line in lines:
      dirs += [line.strip()]
# C routines
ec = ['-fopenmp', '-O2']
el = ['-fopenmp']

setup(name='furryparakeet',
  version='2',
  ext_modules=[Extension('pyimcom_croutines',
    sources = ['pyimcom_croutines.c'],
    include_dirs=dirs,
    extra_compile_args=ec,
    extra_link_args=el
  )],
  py_modules=['pyimcom_croutines', 'pyimcom_lakernel', 'pyimcom_interface']
)
