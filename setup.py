# setup script for pyimcom_croutines

from distutils.core import setup, Extension
import os
from os.path import exists

if __name__ == "__main__":

  # get include directories
  dirs = [];
  if exists('IncludePaths.txt'):
    with open('IncludePaths.txt') as f:
      lines = f.readlines()
      for line in lines:
        dirs += [line.strip()]

  # C routines
  ec = []
  setup(name="pyimcom_croutines",
    version="1",
    ext_modules=[Extension('pyimcom_croutines',
      sources = ['pyimcom_croutines.c'],
      include_dirs=dirs,
      extra_compile_args=ec
    )]
  )

  # Python wrappers
  setup(name='pyimcom_lakernel',
      version='1',
      py_modules=['pyimcom_lakernel'],
  )
  setup(name='pyimcom_interface',
      version='1',
      py_modules=['pyimcom_interface'],
  )
