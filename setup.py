# setup script for pyimcom_croutines

from distutils.core import setup, Extension
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
  setup(name="pyimcom_croutines",
    version="1",
    include_dirs=dirs,
    extra_compile_args = ['-O2'],
    ext_modules=[Extension('pyimcom_croutines', sources = ['pyimcom_croutines.c'])]
  )

  # Python wrappers
  setup(name='pyimcom_lakernel',
      version='1',
      py_modules=['pyimcom_lakernel'],
  )
