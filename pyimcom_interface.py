import numpy
import scipy
import scipy.signal
import scipy.special
import pyimcom_croutines
import pyimcom_lakernel
import time
from astropy.io import fits

print('......3')

##################################################
### Simple PSF models (for testing or outputs) ###
##################################################

# Gaussian spot, n x n, given sigma, centered
# (useful for testing)
def psf_gaussian(n,sigmax,sigmay):
  xa = numpy.linspace((1-n)/2, (n-1)/2, n)
  x = numpy.zeros((n,n))
  y = numpy.zeros((n,n))
  x[:,:] = xa[None,:]
  y[:,:] = xa[:,None]

  I = numpy.exp(-.5*(x**2/sigmax**2+y**2/sigmay**2)) / (2.*numpy.pi*sigmax*sigmay)

  return(I)

# Airy spot, n x n, with lamda/D = ldp pixels,
# and convolved with a tophat (square, full width tophat_conv)
# and Gaussian (sigma)
# and linear obscuration factor (obsc)
#
# result is centered on (n-1)/2,(n-1)/2 (so on a pixel if
# n is odd and a corner if n is even)
#
# normalized to *sum* to unity if analytically extended
#
def psf_simple_airy(n,ldp,obsc=0.,tophat_conv=0.,sigma=0.):

  # figure out pad size -- want to get to at least a tophat width and 6 sigmas
  kp = 1 + int(numpy.ceil(tophat_conv + 6*sigma))
  npad = n + 2*kp

  xa = numpy.linspace((1-npad)/2, (npad-1)/2, npad)
  x = numpy.zeros((npad,npad))
  y = numpy.zeros((npad,npad))
  x[:,:] = xa[None,:]
  y[:,:] = xa[:,None]
  r = numpy.sqrt(x**2+y**2) / ldp # r in units of ldp

  # make Airy spot
  I = (scipy.special.jv(0,numpy.pi*r)+scipy.special.jv(2,numpy.pi*r)
      -obsc**2*(scipy.special.jv(0,numpy.pi*r*obsc)+scipy.special.jv(2,numpy.pi*r*obsc))
      )**2 / (4.*ldp**2*(1-obsc**2)) * numpy.pi

  # now convolve
  It = numpy.fft.fft2(I)
  uxa = numpy.linspace(0,npad-1,npad)/npad
  uxa[-(npad//2):] -= 1
  ux = numpy.zeros((npad,npad))
  uy = numpy.zeros((npad,npad))
  ux[:,:] = uxa[None,:]
  uy[:,:] = uxa[:,None]
  It *= numpy.exp(-2*numpy.pi**2*sigma**2*(ux**2+uy**2)) * numpy.sinc(ux*tophat_conv) * numpy.sinc(uy*tophat_conv)
  I = numpy.real(numpy.fft.ifft2(It))

  return(I[kp:-kp,kp:-kp])

# somewhat messier Airy function with a few diffraction features printed on
# 'features' is an integer that can be added. everything is band limited
def psf_cplx_airy(n,ldp,tophat_conv=0.,sigma=0.,features=0):

  # figure out pad size -- want to get to at least a tophat width and 6 sigmas
  kp = 1 + int(numpy.ceil(tophat_conv + 6*sigma))
  npad = n + 2*kp

  xa = numpy.linspace((1-npad)/2, (npad-1)/2, npad)
  x = numpy.zeros((npad,npad))
  y = numpy.zeros((npad,npad))
  x[:,:] = xa[None,:]
  y[:,:] = xa[:,None]
  r = numpy.sqrt(x**2+y**2) / ldp # r in units of ldp
  phi = numpy.arctan2(y,x)

  # make modified Airy spot
  L1=.8;L2=.01
  f = L1*L2*4./numpy.pi
  II = scipy.special.jv(0,numpy.pi*r)+scipy.special.jv(2,numpy.pi*r)
  for t in range(6):
    II -= f*numpy.sinc(L1*r*numpy.cos(phi+t*numpy.pi/6.))*numpy.sinc(L2*r*numpy.sin(phi+t*numpy.pi/6.))
  I = II**2 / (4.*ldp**2*(1-6*f)) * numpy.pi

  if features%2==1:
    rp = numpy.sqrt((x-1*ldp)**2+(y+2*ldp)**2) / 2. / ldp
    II = scipy.special.jv(0,numpy.pi*rp)+scipy.special.jv(2,numpy.pi*rp)
    I = .8*I + .2*II**2 / (4.*(2.*ldp)**2) * numpy.pi

  if (features//2)%2==1:
    Icopy = numpy.copy(I)
    I *= .85
    I[:-8,:] += .15*Icopy[8:,:]

  if (features//4)%2==1:
    Icopy = numpy.copy(I)
    I *= .8
    I[:-4,:-4] += .1*Icopy[4:,4:]
    I[4:,:-4] += .1*Icopy[:-4,4:]

  # now convolve
  It = numpy.fft.fft2(I)
  uxa = numpy.linspace(0,npad-1,npad)/npad
  uxa[-(npad//2):] -= 1
  ux = numpy.zeros((npad,npad))
  uy = numpy.zeros((npad,npad))
  ux[:,:] = uxa[None,:]
  uy[:,:] = uxa[:,None]
  It *= numpy.exp(-2*numpy.pi**2*sigma**2*(ux**2+uy**2)) * numpy.sinc(ux*tophat_conv) * numpy.sinc(uy*tophat_conv)
  I = numpy.real(numpy.fft.ifft2(It))

  return(I[kp:-kp,kp:-kp])

# rotation matrix by angle theta
# convention is that X(stacking frame) = (distortion_matrix) @ X(native frame)
# in ds9 the native frame is angle theta *clockwise* from stacking frame
# (i.e., theta is position angle of the native frame)
def rotmatrix(theta):
  return(numpy.array([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]]))

# shear matrix by reduced shear (g1,g2)
# unit determinant
def shearmatrix(g1,g2):
  return( numpy.array([[1-g1,-g2],[-g2,1+g1]]) / numpy.sqrt(1.-g1**2-g2**2) )

########################################################
### Now the main functions that we use for real PSFs ###
########################################################

# make PSF overlap class
#
#
class PSF_Overlap:

  # Make the overlap class
  #
  # psf_in_list = list of input PSFs (length n_in)
  # psf_out_list = list of output PSFs (length n_out)
  # dsample = sampling rate of overlap matrices
  # nsample --> sampling matrix has shape (n_tot,n_tot,nsample,nsample) where n_tot = n_in+n_out
  # s_in = input reference pixel scale
  #
  # Note *nsample must be odd*
  #
  # distort_matrices: list of 2x2 distortion matrices associated with the input PSFs
  #   in the form X(stacking frame) = (distortion_matrix) @ X(native frame)
  #
  # amp_penalty = experimental feature to change the weighting of Fourier modes
  #  (do not use or set to None unless you are trying to do a test with it)
  #
  def __init__(self, psf_in_list, psf_out_list, dsample, nsample, s_in, distort_matrices=None, amp_penalty=None):

    # checking
    if nsample%2==0:
      raise Exception('error in PSF_Overlap, nsample={:d} must be odd'.format(nsample))

    self.s_in=s_in

    # number of PSFs
    self.n_in = len(psf_in_list)
    self.n_out = len(psf_out_list)
    self.n_tot = self.n_in + self.n_out
    # save metadata
    self.dsample=dsample
    self.nsample=nsample
    self.nc = nsample//2
    # ... and make an array for the PSF overlaps
    self.overlaparray = numpy.zeros((self.n_tot, self.n_tot, nsample, nsample))

    # make an ns2 x ns2 grid onto which we will interpolate the PSF
    kpad = 5
    ns2 = nsample+2*kpad
    self.psf_array = numpy.zeros((self.n_tot, ns2, ns2))
    xoa = numpy.linspace((1-ns2)/2, (ns2-1)/2, ns2)*dsample
    xo = numpy.zeros((ns2,ns2))
    yo = numpy.zeros((ns2,ns2))
    xo[:,:] = xoa[None,:]
    yo[:,:] = xoa[:,None]
    p = 0 # pad size
    #
    # now the interpolation
    for ipsf in range(self.n_in):

      # get center
      ny = numpy.shape(psf_in_list[ipsf])[0]
      nx = numpy.shape(psf_in_list[ipsf])[1]
      xctr = (nx-1)/2.; yctr = (ny-1)/2.

      detM = 1.
      if distort_matrices is not None:
        if distort_matrices[ipsf] is not None:
          M = numpy.linalg.inv(distort_matrices[ipsf])
          detM = numpy.linalg.det(M)
          xco = M[0,0]*xo + M[0,1]*yo
          yco = M[1,0]*xo + M[1,1]*yo
        else:
          xco = numpy.copy(xo); yco = numpy.copy(yo)
      else:
        xco = numpy.copy(xo); yco = numpy.copy(yo)
      out_array = numpy.zeros((1,ns2*ns2))
      pyimcom_croutines.iD5512C(numpy.pad(psf_in_list[ipsf],p).reshape((1,ny+2*p,nx+2*p)),
        xco.flatten()+xctr+p/2, yco.flatten()+yctr+p/2, out_array)
      self.psf_array[ipsf,:,:] = out_array.reshape((ns2,ns2))

    # ... and the output PSFs
    for ipsf in range(self.n_out):
      # get center
      ny = numpy.shape(psf_out_list[ipsf])[0]
      nx = numpy.shape(psf_out_list[ipsf])[1]
      xctr = (nx-1)/2.; yctr = (ny-1)/2.
      out_array = numpy.zeros((1,ns2*ns2))
      pyimcom_croutines.iD5512C(numpy.pad(psf_out_list[ipsf],p).reshape((1,ny+2*p,nx+2*p)),
        xo.flatten()+xctr+p/2, yo.flatten()+yctr+p/2, out_array)
      self.psf_array[self.n_in+ipsf,:,:] = out_array.reshape((ns2,ns2))

    # FFT based method for overlaps
    nfft = 2**(1+int(numpy.ceil(numpy.log2(ns2-.5))))
    if 3*nfft//4>2*ns2: nfft=3*nfft//4
    b = nsample//2
    psf_array_pad = numpy.zeros((self.n_tot,nfft,nfft))
    psf_array_pad[:,:ns2,:ns2] = self.psf_array
    psf_array_pad_fft = numpy.fft.fft2(psf_array_pad)

    if amp_penalty is not None:
      u = numpy.linspace(0,1.-1/nfft,nfft)
      u = numpy.where(u>.5,u-1,u)
      ut = numpy.sqrt(u[:,None]**2+u[None,:]**2)
      for ip in range(self.n_tot): psf_array_pad_fft[ip,:,:] *= 1. + amp_penalty['amp']*numpy.exp(-2*numpy.pi**2*ut**2*amp_penalty['sig']**2)

    # ... continue FFT
    for ipsf in range(self.n_tot):
      for jpsf in range(0,ipsf+1,2):
        if ipsf==jpsf:
          ift = numpy.fft.ifftshift(numpy.fft.ifft2(psf_array_pad_fft[ipsf,:,:]*
            numpy.conjugate(psf_array_pad_fft[jpsf,:,:])))
          self.overlaparray[ipsf,jpsf,:,:] = numpy.real(ift[nfft//2-b:nfft//2+b+1,nfft//2-b:nfft//2+b+1])
        else:
          ift = numpy.fft.ifftshift(numpy.fft.ifft2(psf_array_pad_fft[ipsf,:,:]*
            numpy.conjugate(psf_array_pad_fft[jpsf,:,:]-1j*psf_array_pad_fft[jpsf+1,:,:])))
          self.overlaparray[ipsf,jpsf,:,:] = numpy.real(ift[nfft//2-b:nfft//2+b+1,nfft//2-b:nfft//2+b+1])
          self.overlaparray[ipsf,jpsf+1,:,:] = numpy.imag(ift[nfft//2-b:nfft//2+b+1,nfft//2-b:nfft//2+b+1])

    # the 'above diagonal' can be obtained by flipping
    for ipsf in range(1,self.n_tot):
      for jpsf in range(ipsf):
        self.overlaparray[jpsf,ipsf,:,:] = self.overlaparray[ipsf,jpsf,::-1,::-1]

  # ### <-- end __init__ here ###

# Function to go from PSF overlap object to transfer matrices.
#
# Inputs:
#   psfobj = PSF overlap class object
#   psf_oversamp_factor = PSF oversampling factor relative to native pixel scale (float)
#   targetleak = target leakage, shape=(n_out,) or length n_out list
#   ctrpos = list (length n_in) of postage stamp centroids in stacking frame, shape=(2,) ** (x,y) format **
#   distort_matrices = list (length n_in) of shape=(2,2) matrices
#   in_stamp_dscale = input postage stamp scale
#   in_stamp_shape = input postage stamp size, length 2 tuple (ny_in,nx_in)
#   out_stamp_dscale = output postage stamp scale
#   out_stamp_shape = output postage stamp size, length 2 tuple (ny_out,nx_out)
#   in_mask = boolean mask, 'True' means a good pixel. shape = (n_in,ny_in,nx_in) [set to None to accept all]
#   tbdy_radius = radius of boundary to clip pixels in input images
#   smax = maximum allowed value of Sigma (1 if you want to avoid amplifying noise)
#   flat_penalty = amount by which to penalize having different contributions to the output from different input images
#
# Outputs: dictionary containing:
#   Sigma = output noise amplification, shape=(n_out,ny_out,nx_out)
#   UC = fractional squared error in PSF, shape=(n_out,ny_out,nx_out)
#   T = coaddition matrix, shape=(n_out,ny_out,nx_out,n_in,ny_in,nx_in)
#       ... and the intermediate results kappa, A, mBhalf, C, fullmask
#
def get_coadd_matrix(psfobj, psf_oversamp_factor, targetleak, ctrpos, distort_matrices,
  in_stamp_dscale, in_stamp_shape, out_stamp_dscale, out_stamp_shape, in_mask, tbdy_radius, smax=1., flat_penalty=0.):

  # number of input and output images
  n_in = psfobj.n_in
  n_out = psfobj.n_out

  # get information
  (ny_in,nx_in) = in_stamp_shape
  (ny_out,nx_out) = out_stamp_shape

  # positions of the pixels
  xpos = numpy.zeros((n_in,ny_in,nx_in))
  ypos = numpy.zeros((n_in,ny_in,nx_in))
  for i in range(n_in):
    xo = numpy.zeros((ny_in,nx_in))
    yo = numpy.zeros((ny_in,nx_in))
    xo[:,:] = numpy.linspace((1-nx_in)/2, (nx_in-1)/2, nx_in)[None,:]
    yo[:,:] = numpy.linspace((1-ny_in)/2, (ny_in-1)/2, ny_in)[:,None]
    xpos[i,:,:] = in_stamp_dscale*(distort_matrices[i][0,0]*xo + distort_matrices[i][0,1]*yo) + ctrpos[i][0]
    ypos[i,:,:] = in_stamp_dscale*(distort_matrices[i][1,0]*xo + distort_matrices[i][1,1]*yo) + ctrpos[i][1]

  # mask information and table
  if in_mask is None:
    full_mask = numpy.full((n_in,ny_in,nx_in), True)
  else:
    full_mask = numpy.copy(in_mask)
  # now do the radius clipping
  Lx = out_stamp_dscale*(nx_out-1)/2.
  Ly = out_stamp_dscale*(ny_out-1)/2.
  full_mask = numpy.where( numpy.maximum(numpy.abs(xpos)-Lx,0.)**2 + numpy.maximum(numpy.abs(ypos)-Ly,0.)**2<tbdy_radius**2, full_mask, False)
  masklayers = []
  #
  # ... and we will make a list version of the mask.
  # ngood[i] pixels to be used from i th input image
  # ... starting from position nstart[i]
  #     (formally: nstart[n_in] == n_all)
  # n_all total
  #
  ngood = numpy.zeros((n_in,),numpy.int32)
  nstart = numpy.zeros((n_in+1,),numpy.int32)
  for i in range(n_in):
    masklayers += [numpy.where(full_mask[i,:,:])]
    ngood[i] = len(masklayers[i][0])
    nstart[i+1:] += ngood[i]
  n_all = numpy.sum(ngood)

  # and build list of positions
  my_x = []; my_y = []
  for i in range(n_in):
    my_x += [xpos[i,masklayers[i][0],masklayers[i][1]].flatten()]
    my_y += [ypos[i,masklayers[i][0],masklayers[i][1]].flatten()]

  # build optimization matrices
  A = numpy.zeros((n_all,n_all))
  mBhalf = numpy.zeros((n_out, nx_out*ny_out, n_all))
  C = numpy.zeros((n_out,))
  #
  # the A-matrix first
  s = psfobj.dsample/psf_oversamp_factor*psfobj.s_in
  for i in range(n_in):
    for j in range(n_in):
      if j>=i:
        ddx = numpy.zeros((ngood[i],ngood[j]))
        ddy = numpy.zeros((ngood[i],ngood[j]))
        ddx[:,:] = my_x[i][:,None] - my_x[j][None,:]
        ddy[:,:] = my_y[i][:,None] - my_y[j][None,:]
        out1 = numpy.zeros((1,ngood[i]*ngood[j]))
        pyimcom_croutines.iD5512C(psfobj.overlaparray[i,j,:,:].reshape((1,psfobj.nsample,psfobj.nsample)),
          ddx.flatten()/s+psfobj.nc, ddy.flatten()/s+psfobj.nc, out1)
        A[nstart[i]:nstart[i+1],nstart[j]:nstart[j+1]] = out1.reshape((ngood[i],ngood[j]))

      else:
        # we have already computed this component of A
        A[nstart[i]:nstart[i+1],nstart[j]:nstart[j+1]] = A[nstart[j]:nstart[j+1],nstart[i]:nstart[i+1]].T
  #
  # flat penalty
  if flat_penalty>0.:
    for i in range(n_in):
      for j in range(n_in):
        A[nstart[i]:nstart[i+1],nstart[j]:nstart[j+1]] -= flat_penalty / n_in / 2.
        A[nstart[j]:nstart[j+1],nstart[i]:nstart[i+1]] -= flat_penalty / n_in / 2.
      A[nstart[i]:nstart[i+1],nstart[i]:nstart[i+1]] += flat_penalty

  # force exact symmetry
  A = (A+A.T)/2.
  # end A matrix

  # now the mBhalf matrix
  # first get output pixel positions
  xout = numpy.zeros((ny_out,nx_out))
  yout = numpy.zeros((ny_out,nx_out))
  xout[:,:] = numpy.linspace((1-nx_out)/2, (nx_out-1)/2, nx_out)[None,:] * out_stamp_dscale
  yout[:,:] = numpy.linspace((1-ny_out)/2, (ny_out-1)/2, ny_out)[:,None] * out_stamp_dscale
  xout = xout.flatten()
  yout = yout.flatten()
  for i in range(n_in):
    for j in range(n_out):
      ddx = numpy.zeros((ngood[i],ny_out*nx_out))
      ddy = numpy.zeros((ngood[i],ny_out*nx_out))
      ddx[:,:] = my_x[i][:,None] - xout[None,:]
      ddy[:,:] = my_y[i][:,None] - yout[None,:]
      out1 = numpy.zeros((1,ngood[i]*ny_out*nx_out))
      pyimcom_croutines.iD5512C(psfobj.overlaparray[i,psfobj.n_in+j,:,:].reshape((1,psfobj.nsample,psfobj.nsample)),
        ddx.flatten()/s+psfobj.nc, ddy.flatten()/s+psfobj.nc, out1)
      mBhalf[j,:,nstart[i]:nstart[i+1]] = out1.reshape((ngood[i],ny_out*nx_out)).T

  # and C
  C = numpy.zeros((n_out,))
  for j in range(n_out): C[j] = psfobj.overlaparray[n_in+j,n_in+j,psfobj.nc,psfobj.nc]

  # generate matrices
  (kappa_, Sigma_, UC_, T_) = pyimcom_lakernel.BruteForceKernel(A,mBhalf,C,numpy.array(targetleak),smax=smax)

  # this code was just for testing, only works for n_out = 1
  # (kappa_, Sigma_, UC_, T_) = pyimcom_lakernel.BruteForceKernel(A,mBhalf[0,:,:],C[0],targetleak[0])
  # T_ = T_.reshape(1,ny_out*nx_out,nstart[n_in])

  # post processing
  kappa = kappa_.reshape((n_out,ny_out,nx_out))
  Sigma = Sigma_.reshape((n_out,ny_out,nx_out))
  UC = UC_.reshape((n_out,ny_out,nx_out))
  #
  # for T: (nt,m,n) --> (n_out,ny_out,nx_out,n_in,ny_in,nx_in)
  T = numpy.zeros((n_out,ny_out,nx_out,n_in,ny_in,nx_in))
  for i in range(n_in):
    for k in range(ngood[i]):
      T[:,:,:,i,masklayers[i][0][k],masklayers[i][1][k]] = T_[:,:,nstart[i]+k].reshape((n_out,ny_out,nx_out))

  return {
    'Sigma': Sigma,
    'UC': UC,
    'T': T,
    'kappa': kappa,
    'A': A,
    'mBhalf': mBhalf,
    'C': C,
    'full_mask': full_mask
  }

# Function to creat input postage stamps of point sources of unit flux and coadd them to test the PSF matrices.
#
# Inputs:
#   psf_in_list = list of input PSFs (length n_in)
#   psf_out_list = list of output PSFs (length n_out)
#   psf_oversamp_factor = PSF oversampling factor relative to native pixel scale (float)
#   ctrpos = list (length n_in) of postage stamp centroids in stacking frame, shape=(2,) ** (x,y) format **
#   distort_matrices = list (length n_in) of shape=(2,2) matrices
#   T = coaddition matrix, shape=(n_out,ny_out,nx_out,n_in,ny_in,nx_in)
#   in_mask = boolean mask, 'True' means a good pixel. shape = (n_in,ny_in,nx_in) [set to None to accept all]
#   in_stamp_dscale = input (native) plate scale
#   out_stamp_dscale = output postage stamp scale
#   srcpos = position of the point source to inject, shape=(2,) ** (x,y) format **
#
# Outputs:
#   input postage stamps, shape=(n_in,ny_in,nx_in)
#   output postage stamp, shape=(n_out,ny_out,nx_out)
#   output postage stamp error, shape=(n_out,ny_out,nx_out)
#
def test_psf_inject(psf_in_list, psf_out_list, psf_oversamp_factor, ctrpos, distort_matrices, T, in_mask, in_stamp_dscale, out_stamp_dscale, srcpos):

  # basic info
  (n_out,ny_out,nx_out,n_in,ny_in,nx_in) = numpy.shape(T)

  # center pixel of input stamps
  xctr = (nx_in-1)/2.; yctr = (ny_in-1)/2.

  # make input stamp array
  in_array = numpy.zeros((n_in,ny_in,nx_in))

  p = 5 # pad length

  # make the input stamps
  for ipsf in range(n_in):
    (ny,nx) = numpy.shape(psf_in_list[ipsf])

    # get position of source in stamp coordinates
    xpsf = srcpos[0] - ctrpos[ipsf][0]
    ypsf = srcpos[1] - ctrpos[ipsf][1]
    M = numpy.linalg.inv(in_stamp_dscale*distort_matrices[ipsf])
    xpos = M[0,0]*xpsf + M[0,1]*ypsf + xctr
    ypos = M[1,0]*xpsf + M[1,1]*ypsf + yctr

    # now pixel positions relative to the PSF
    inX = numpy.zeros((ny_in,nx_in))
    inY = numpy.zeros((ny_in,nx_in))
    inX[:,:] = numpy.linspace(-xpos,nx_in-1-xpos,nx_in)[None,:]
    inY[:,:] = numpy.linspace(-ypos,ny_in-1-ypos,ny_in)[:,None]
    interp_array = numpy.zeros((1,ny_in*nx_in))
    pyimcom_croutines.iD5512C(numpy.pad(psf_in_list[ipsf],p).reshape((1,ny+2*p,nx+2*p)),
      psf_oversamp_factor*inX.flatten() + (nx-1)/2.+p, psf_oversamp_factor*inY.flatten() + (ny-1)/2.+p, interp_array)
    in_array[ipsf,:,:] = interp_array.reshape((ny_in,nx_in)) * psf_oversamp_factor**2

  if in_mask is not None:
    in_array = numpy.where(in_mask, in_array, 0.)

  # --- end construction of the input postage stamps ---

  out_array = (T.reshape(n_out*ny_out*nx_out,n_in*ny_in*nx_in)@in_array.flatten()).reshape(n_out,ny_out,nx_out)

  # --- and now the 'target' output array ---
  target_out_array = numpy.zeros((n_out,ny_out,nx_out))
  xctr = (nx_out-1)/2.; yctr = (ny_out-1)/2.
  for ipsf in range(n_out):
    (ny,nx) = numpy.shape(psf_out_list[ipsf])

    # get position of source in stamp coordinates
    xpos = srcpos[0] / out_stamp_dscale + xctr
    ypos = srcpos[1] / out_stamp_dscale + yctr

    # now pixel positions relative to the PSF
    inX = numpy.zeros((ny_out,nx_out))
    inY = numpy.zeros((ny_out,nx_out))
    inX[:,:] = numpy.linspace(-xpos,nx_out-1-xpos,nx_out)[None,:] * out_stamp_dscale/in_stamp_dscale
    inY[:,:] = numpy.linspace(-ypos,ny_out-1-ypos,ny_out)[:,None] * out_stamp_dscale/in_stamp_dscale
    interp_array = numpy.zeros((1,ny_out*nx_out))
    pyimcom_croutines.iD5512C(numpy.pad(psf_out_list[ipsf],p).reshape((1,ny+2*p,nx+2*p)),
      psf_oversamp_factor*inX.flatten() + (nx-1)/2.+p, psf_oversamp_factor*inY.flatten() + (ny-1)/2.+p, interp_array)
    target_out_array[ipsf,:,:] = interp_array.reshape((ny_out,nx_out)) * psf_oversamp_factor**2

  return(in_array,out_array,out_array-target_out_array)

#############################
### Functions for testing ###
#############################

# simple test for the airy function
def testairy():
  IA = psf_simple_airy(128,4,tophat_conv=4,sigma=4*.3)
  print(numpy.sum(IA))
  hdu = fits.PrimaryHDU(IA)
  hdu.writeto('testairy.fits', overwrite=True)

# simple test for the PSF overlap function
def testpsfoverlap():
  n1 = 256
  ld = 1.29/2.37e6*206265/.11
  nps=8
  cd=.3
  Im1 = psf_simple_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd)[4:,4:]
  Im2 = psf_cplx_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd)
  Im3 = psf_cplx_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd)
  Im4 = psf_cplx_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd)
  Im5 = psf_cplx_airy(n1,nps*ld,tophat_conv=nps,sigma=nps*cd)
  ImOut = psf_simple_airy(n1,nps*ld,tophat_conv=nps,sigma=6.)

  s_out=.04
  s_in=.11

  t1a = time.perf_counter()
  P = PSF_Overlap([Im1,Im2,Im3,Im4,Im5], [ImOut], .5, 511, s_in,
    distort_matrices=[rotmatrix(0.), rotmatrix(numpy.pi/4.), rotmatrix(numpy.pi/3.), None, None])
  t1b = time.perf_counter()
  print('timing psf overlap: ', t1b-t1a)

  hdu = fits.PrimaryHDU(P.psf_array)
  hdu.writeto('testpsf1.fits', overwrite=True)
  hdu = fits.PrimaryHDU(numpy.transpose(P.overlaparray,axes=(0,2,1,3)).reshape(P.nsample*P.n_tot,P.nsample*P.n_tot))
  hdu.writeto('testpsf2.fits', overwrite=True)

  # and now make a coadd matrix
  t1c = time.perf_counter()
  get_coadd_matrix(P, float(nps), [1.e-8], [(0.055,0), (0,0), (0,0.025), (0,0), (0,0.055)],
    [rotmatrix(0.), rotmatrix(numpy.pi/4.), rotmatrix(numpy.pi/3.), rotmatrix(0.), rotmatrix(0.)],
    s_in, (42,42), s_out, (27,27), None, .66)
  t1d = time.perf_counter()
  print('timing coadd matrix: ', t1d-t1c)

# only for testing purposes
#testairy()
#testpsfoverlap()
