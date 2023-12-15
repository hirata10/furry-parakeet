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
n_out        = 1 # number of output images
nps          = 8 # PSF oversampling factor
s_in         = .11 # input pixel scale in arcsec
s_out        = .025 # output pixel scale in arcsec
cd           = .3 # charge diffusion, rms per axis in pixels
n1           = 512 # PSF postage stamp size
extbdy       = 1. # boundary extension in arcsec
uctarget     = 2e-4 # target image leakage (in RMS units)
flat_penalty = 1e-8 # penalty for having different dependences on different inputs

# default parameters
seed         = 1000 # default seed
pdtype       = 'R' # positional dither type (R=random, D=deterministic)
sigout       = numpy.sqrt(1./12.+cd**2) # output smoothing
roll         = None # no rolls
shear        = None # no camera shear
magnify      = None # no camera magnification
badfrac      = 0. # fraction of pixels to randomly kill
messyPSF     = False # put messy asymmetries in the PSF

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

  m = re.search(r'^SHEAR:', line) # shears g1,g2 for each exposure
  if m:
    shear = line.split()[1:]
    for j in range(len(shear)): shear[j] = float(shear[j])

  m = re.search(r'^MAGNIFY:', line) # plate scale divided by 1+magnify[], so >0 --> smaller plate scale --> magnified image
  if m:
    magnify = line.split()[1:]
    for j in range(len(magnify)): magnify[j] = float(magnify[j])

  m = re.search(r'^MESSYPSF', line)
  if m: messyPSF = True

rng = numpy.random.default_rng(seed)

print('lambda =', lam, 'micron')
print('n_in =', n_in)

print('output --> ', outstem)

ld = lam/2.37e6*206265./s_in

ImIn = []
mlist = []
posoffset = []
if roll is None: roll = numpy.zeros((n_in,)).tolist()
if shear is None: shear = numpy.zeros((2*n_in,)).tolist()
if magnify is None: magnify = numpy.zeros((n_in,)).tolist()
for k in range(n_in):
  fk = 0
  if messyPSF: fk=k
  ImIn += [ pyimcom_interface.psf_cplx_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd,features=fk) ]
  mlist += [pyimcom_interface.rotmatrix(roll[k])@pyimcom_interface.shearmatrix(shear[2*k],shear[2*k+1])/(1.+magnify[k])]

  # positional offsets
  f = numpy.zeros((2,))
  f[0] = rng.random()
  f[1] = rng.random()
  Mf = s_in*mlist[k]@f
  posoffset += [(Mf[0],Mf[1])]
ImOut = [ pyimcom_interface.psf_simple_airy(n1,nps*ld,tophat_conv=0.,sigma=nps*sigout) ]

print('translation:', posoffset)
print('roll:', numpy.array(roll)*180./numpy.pi)
print('magnify:', numpy.array(magnify))
print('shear g1:', numpy.array(shear)[::2])
print('shear g2:', numpy.array(shear)[1::2])

t1a = time.perf_counter()
P = pyimcom_interface.PSF_Overlap(ImIn, ImOut, .5, 2*n1-1, s_in, distort_matrices=mlist)
t1b = time.perf_counter()
print('timing psf overlap: ', t1b-t1a)

hdu = fits.PrimaryHDU(P.psf_array)
hdu.writeto(outstem+'testpsf1.fits', overwrite=True)
q1 = 48
if q1>=n1: q1=n1-1
q2 = 2*q1+1
hdu = fits.PrimaryHDU(numpy.transpose(P.psf_array[:,n1-q1:n1+q1+1,n1-q1:n1+q1+1],axes=(1,0,2)).reshape(q2,q2*(n_in+n_out)))
hdu.writeto(outstem+'testpsf1b.fits', overwrite=True)
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
ims = pyimcom_interface.get_coadd_matrix(P, float(nps), [uctarget**2], posoffset, mlist, s_in, (ny_in,nx_in), s_out, (ny_out,nx_out), inmask, extbdy, smax=1./n_in, flat_penalty=flat_penalty, choose_outputs='ABCKMSTU')
t1d = time.perf_counter()
print('timing coadd matrix: ', t1d-t1c)

print('number of output x input pixels used:', numpy.shape(ims['mBhalf']))

print('C = ', ims['C'])
hdu = fits.PrimaryHDU(numpy.where(ims['full_mask'],1,0).astype(numpy.uint16)); hdu.writeto(outstem+'mask.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['A']); hdu.writeto(outstem+'A.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['mBhalf']); hdu.writeto(outstem+'mBhalf.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['C']); hdu.writeto(outstem+'C.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['kappa']); hdu.writeto(outstem+'kappa.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.sqrt(ims['Sigma'])); hdu.writeto(outstem+'sqSigma.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.sqrt(ims['UC'])); hdu.writeto(outstem+'sqUC.fits', overwrite=True)
hdu = fits.PrimaryHDU(ims['T'].reshape((n_out,ny_out*nx_out, n_in*ny_in*nx_in,))); hdu.writeto(outstem+'T.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.transpose(numpy.sum(ims['T'], axis=(4,5)), axes=(1,0,3,2)).reshape((ny_out,nx_out*n_in*n_out))); hdu.writeto(outstem+'Tsum.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.where(ims['full_mask'],1,0).astype(numpy.uint16)); hdu.writeto(outstem+'mask.fits', overwrite=True)

# make test image
# put test source in lower-right quadrant as displayed in ds9
#
# [   |   ]
# [   |   ]
# [   |   ]   (60% of the way from left to right,
# [---+---]    20% of the way from bottom to top)
# [   |   ]
# [   |X  ]
# [   |   ]
#
test_srcpos = (.1*(nx_out-1)*s_out, -.3*(ny_out-1)*s_out)
(intest, outtest, outerr) = pyimcom_interface.test_psf_inject(ImIn, ImOut, nps, posoffset, mlist, ims['T'], inmask, s_in, s_out, test_srcpos)
print('input image sums =', numpy.sum(intest, axis=(1,2)))
hdu = fits.PrimaryHDU(intest); hdu.writeto(outstem+'sample_ptsrc_in.fits', overwrite=True)
hdu = fits.PrimaryHDU(numpy.transpose(intest, axes=(1,0,2)).reshape(ny_in,n_in*nx_in)); hdu.writeto(outstem+'sample_ptsrc_in_flat.fits', overwrite=True)
hdu = fits.PrimaryHDU(outtest); hdu.writeto(outstem+'sample_ptsrc_out.fits', overwrite=True)
hdu = fits.PrimaryHDU(outerr); hdu.writeto(outstem+'sample_ptsrc_out_err.fits', overwrite=True)
amp = numpy.zeros((n_out,))
for ipsf in range(n_out):
  print('error {:2d} {:11.5E}'.format(ipsf, numpy.sqrt(numpy.sum(outerr[ipsf,:,:]**2)/numpy.sum(outtest[ipsf,:,:]**2))))
  amp[ipsf] = numpy.sqrt(numpy.sum(outtest[ipsf,:,:]**2))

# test input with Gaussian white noise
noise_in = rng.normal(0., 1., size=(n_in*ny_in*nx_in,))
noise_out = (ims['T'].reshape((n_out,ny_out*nx_out, n_in*ny_in*nx_in))@noise_in).reshape((n_out, ny_out, nx_out))
hdu = fits.PrimaryHDU(numpy.transpose(noise_out, axes=(1,0,2)).reshape(ny_out, n_out*nx_out)); hdu.writeto(outstem+'samplenoise.fits', overwrite=True)

# more tests with random pt src positions -- to do statistics on how well things worked in this stamp
print('')
allerr = []
for itest in range(1000):
  test_srcpos = ((s_out*(nx_out-1)*.5+extbdy)*(rng.random()-.5), (s_out*(ny_out-1)*.5+extbdy)*(rng.random()-.5))
  #print('-- random point {:3d} at ({:8.5f},{:8.5f}) --'.format(itest, test_srcpos[0], test_srcpos[1]))
  (intest, outtest, outerr) = pyimcom_interface.test_psf_inject(ImIn, ImOut, nps, posoffset, mlist, ims['T'], inmask, s_in, s_out, test_srcpos)
  #for ipsf in range(n_out):
  #  print('error {:3d},{:2d} {:11.5E}    :'.format(itest, ipsf, numpy.sqrt(numpy.sum(outerr[ipsf,:,:]**2))/amp[ipsf]))
  allerr += [numpy.sqrt(numpy.sum(outerr[ipsf,:,:]**2))/amp[ipsf]]
allerr = numpy.array(allerr)
print('rms err = {:11.5E}'.format(numpy.sqrt(numpy.mean(allerr**2))))
print('med err = {:11.5E}'.format(numpy.median(allerr)))
print('max err = {:11.5E}'.format(numpy.amax(allerr)))

# print summary information
print('percentiles of sqrtUC, sqrtSigma, kappa:')
for i in [1,2,5,10,25,50,75,90,95,98,99]:
  print('    {:2d}% {:11.5E} {:11.5E} {:11.5E}'.format(i, numpy.percentile(numpy.sqrt(ims['UC']),i), numpy.percentile(numpy.sqrt(ims['Sigma']),i),
    numpy.percentile(ims['kappa'],i) ))

print('kappa range = {:11.5E}, {:11.5E}'.format(numpy.amin(ims['kappa']), numpy.amax(ims['kappa'])))
