import numpy
import sys
import pyimcom_interface
import time
import re
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Read config file
config_file = sys.argv[1]
with open(config_file) as myf: content = myf.read().splitlines()

# basic parameters
n_out      = 1 # number of output images
nps        = 8 # PSF oversampling factor
s_in       = .11 # input pixel scale in arcsec
s_out      = .04 # output pixel scale in arcsec
cd         = .3 # charge diffusion, rms per axis in pixels
n1         = 256 # PSF postage stamp size
extbdy     = .88 # boundary extension in arcsec
uctarget   = 1e-4 # target image leakage (in RMS units)

# default parameters
seed       = 1000 # default seed
pdtype     = 'R' # positional dither type (R=random, D=deterministic)
sigout     = numpy.sqrt(1./12.+cd**2) # output smoothing
roll       = None # no rolls
badfrac    = 0. # fraction of pixels to randomly kill

for line in content:

  # -- REQUIRED KEYWORDS --

  m = re.search(r'^LAMBDA:\s*(\S+)', line)
  if m: lam = float(m.group(1))

  m = re.search(r'^N:\s*(\d+)', line)
  if m: n_in = int(m.group(1))

  m = re.search(r'^OUT:\s*(\S+)', line)
  if m: outstem = m.group(1)

  m = re.search(r'^OUTSIZE:\s*(\d+)\s+(\d+)', line) # ny, then nx
  if m:
    ny_out = int(m.group(1))
    nx_out = int(m.group(2))

  m = re.search(r'^INSIZE:\s*(\d+)\s+(\d+)', line) # ny, then nx
  if m:
    ny_in = int(m.group(1))
    nx_in = int(m.group(2))

  # -- OPTIONS --
  m = re.search(r'^RNGSEED:\s*(\d+)', line)  # RNG seed
  if m: seed=int(m.group(1))

  m = re.search(r'^EXTRASMOOTH:\s*(\S+)', line) # extra smoothing, pix rms per axis
  if m: sigout=numpy.sqrt(sigout**2+float(m.group(1))**2)

  m = re.search(r'^UCTARGET:\s*(\S+)', line)
  if m: uctarget=float(m.group(1))

  m = re.search(r'^BADFRAC:\s*(\S+)', line)
  if m: badfrac=float(m.group(1))

  m = re.search(r'^ROLL:', line) # input is in degrees
  if m:
    roll = line.split()[1:]
    for j in range(len(roll)): roll[j] = float(roll[j])*numpy.pi/180. # convert to float

rng = numpy.random.default_rng(seed)

print('lambda =', lam, 'micron')
print('n_in =', n_in)

print('output --> ', outstem)

ld = lam/2.37e6*206265./s_in

ImIn = []
mlist = []
mlist2 = []
posoffset = []
if roll is None: roll = numpy.zeros((n_in,)).tolist()
for k in range(n_in):
  ImIn += [ pyimcom_interface.psf_cplx_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd) ]
  mlist += [pyimcom_interface.rotmatrix(roll[k])]
  mlist2 += [pyimcom_interface.rotmatrix(roll[k])*s_in]

  # positional offsets
  f = numpy.zeros((2,))
  f[0] = rng.random()
  f[1] = rng.random()
  Mf = mlist2[k]@f
  posoffset += [(Mf[0],Mf[1])]
ImOut = [ pyimcom_interface.psf_simple_airy(n1,nps*ld,tophat_conv=0.,sigma=nps*sigout) ]

print('translation:', posoffset)
print('roll:', numpy.array(roll)*180./numpy.pi)

t1a = time.perf_counter()
P = pyimcom_interface.PSF_Overlap(ImIn, ImOut, .5, 2*n1-1, s_in, distort_matrices=mlist)
t1b = time.perf_counter()
print('timing psf overlap: ', t1b-t1a)

hdu = fits.PrimaryHDU(P.psf_array)
hdu.writeto(outstem+'testpsf1.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.transpose(P.overlaparray,axes=(0,2,1,3)).reshape(P.nsample*P.n_tot,P.nsample*P.n_tot))
hdu.writeto(outstem+'testpsf2.fits', overwrite=True)

# mask
inmask=None
if badfrac>0:
  inmask = numpy.where(rng.random(size=(n_in,ny_in,nx_in))>badfrac,True,False)
  print('number good', numpy.count_nonzero(inmask), 'of', n_in*ny_in*nx_in)
  print(numpy.shape(inmask))

# and now make a coadd matrix
t1c = time.perf_counter()
ims = pyimcom_interface.get_coadd_matrix(P, float(nps), [uctarget**2], posoffset, mlist2, (ny_in,nx_in), s_out, (ny_out,nx_out), inmask, extbdy)
t1d = time.perf_counter()
print('timing coadd matrix: ', t1d-t1c)

print('number of output x input pixels used:', numpy.shape(ims['mBhalf']))

print('C = ', ims['C'])
hdu = fits.PrimaryHDU(numpy.where(ims['full_mask'],1,0).astype(numpy.uint16)); hdu.writeto('mask.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['A']); hdu.writeto(outstem+'A.fits', overwrite=True)
hdu = fits.PrimaryHDU((ims['A']-ims['A'].T)/2); hdu.writeto(outstem+'A_asym.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['mBhalf']); hdu.writeto(outstem+'mBhalf.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['C']); hdu.writeto(outstem+'C.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['kappa']); hdu.writeto(outstem+'kappa.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.sqrt(ims['Sigma'])); hdu.writeto(outstem+'sqSigma.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.sqrt(ims['UC'])); hdu.writeto(outstem+'sqUC.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['T'].reshape((n_out,ny_out*nx_out, n_in*ny_in*nx_in,))); hdu.writeto(outstem+'T.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.where(ims['full_mask'],1,0).astype(numpy.uint16)); hdu.writeto(outstem+'mask.fits', overwrite=True)

# print summary information
print('percentiles of sqrtUC, sqrtSigma, kappa:')
for i in [1,2,5,10,25,50,75,90,95,98,99]:
  print('    {:2d}% {:11.5E} {:11.5E} {:11.5E}'.format(i, numpy.percentile(numpy.sqrt(ims['UC']),i), numpy.percentile(numpy.sqrt(ims['Sigma']),i),
    numpy.percentile(ims['kappa'],i) ))
