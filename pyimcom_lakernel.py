import numpy
import scipy
import pyimcom_croutines
import time
import warnings

# 'Brute force' version of the kernel
# Slow and useful only for comparisons
#
# Inputs:
#   A = system matrix, shape=(n,n)
#   mBhalf = -B/2 = target overlap matrix, shape=(m,n)
#   C = target normalization (scalar)
#   targetleak = allowable leakage of target PSF
#   kCmin, kCmax, nbis = range of kappa/C to test, number of bisections
#
# Outputs:
#   kappa = Lagrange multiplier per output pixel, shape=(m,)
#   Sigma = output noise amplification, shape=(m,)
#   UC = fractional squared error in PSF, shape=(m,)
#   T = coaddition matrix, shape=(m,n)
#
def BruteForceKernel(A,mBhalf,C,targetleak,kCmin=1e-16,kCmax=1e16,nbis=53):

  # get dimensions
  (m,n) = numpy.shape(mBhalf)

  # eigensystem
  lam, Q = numpy.linalg.eigh(A)
  # -P/2 matrix
  mPhalf = mBhalf@Q

  # allocate targets
  kappa = numpy.zeros((m,))
  T = numpy.zeros((m,n))
  UC = numpy.zeros((m,))
  Sigma = numpy.zeros((m,))

  # now loop over pixels
  for a in range(m):
    factor = numpy.sqrt(kCmax/kCmin)
    kappa[a] = numpy.sqrt(kCmax*kCmin)
    for ibis in range(nbis+1):
      factor = numpy.sqrt(factor)
      UC[a] = 1-numpy.sum((lam+2*kappa[a])/(lam+kappa[a])**2*mPhalf[a,:]**2)/C
      if ibis!=nbis:
        if UC[a]>targetleak:
          kappa[a] /= factor
        else:
          kappa[a] *= factor
    T[a,:] = Q@(mPhalf[a,:]/(lam+kappa[a]))
    Sigma[a] = numpy.sum((mPhalf[a,:]/(lam+kappa[a]))**2)

  return (kappa,Sigma,UC,T)

#
# ... and the same function but wrapped around C:
def CKernel(A,mBhalf,C,targetleak,kCmin=1e-16,kCmax=1e16,nbis=53):

  # get dimensions
  (m,n) = numpy.shape(mBhalf)

  # eigensystem
  lam, Q = numpy.linalg.eigh(A)
  # -P/2 matrix
  mPhalf = mBhalf@Q

  # output arrays
  kappa = numpy.zeros((m,))
  Sigma = numpy.zeros((m,))
  UC = numpy.zeros((m,))
  tt = numpy.zeros((m,n))

  pyimcom_croutines.lakernel1(lam,Q,mPhalf,C,targetleak,kCmin,kCmax,nbis,kappa,Sigma,UC,tt,1e49)
  T = tt@Q.T
  return (kappa,Sigma,UC,T)

# This one generates multiple images. there can be nt target PSFs.
# if 2D arrays are input then assumes nt=1
#
# Inputs:
#   A = system matrix, shape=(n,n)
#   mBhalf = -B/2 = target overlap matrix, shape=(nt,m,n)
#   C = target normalization, shape = (nt,)
#   targetleak = allowable leakage of target PSF (nt,)
#   kCmin, kCmax, nbis = range of kappa/C to test, number of bisections
#   smax = maximum allowed Sigma
#
# Outputs:
#   kappa = Lagrange multiplier per output pixel, shape=(nt,m)
#   Sigma = output noise amplification, shape=(nt,m)
#   UC = fractional squared error in PSF, shape=(nt,m)
#   T = coaddition matrix, shape=(nt,m,n)
#
def CKernelMulti(A,mBhalf,C,targetleak,kCmin=1e-16,kCmax=1e16,nbis=53,smax=1e8):

  # eigensystem
  lam, Q = numpy.linalg.eigh(A)

  # get dimensions and mPhalf matrix
  if mBhalf.ndim==2:
    nt=1
    (m,n) = numpy.shape(mBhalf)
    mBhalf_image = mBhalf.reshape((1,m,n))
    C_s = numpy.array([C])
    targetleak_s = numpy.array([targetleak])
  else:
    (nt,m,n) = numpy.shape(mBhalf)
    mBhalf_image = mBhalf
    C_s = C
    targetleak_s = targetleak

  # output arrays
  kappa = numpy.zeros((nt,m))
  Sigma = numpy.zeros((nt,m))
  UC = numpy.zeros((nt,m))
  T = numpy.zeros((nt,m,n))
  tt = numpy.zeros((m,n))

  for k in range(nt):
    pyimcom_croutines.lakernel1(lam,Q,mBhalf_image[k,:,:]@Q,C_s[k],targetleak_s[k],kCmin,kCmax,nbis,kappa[k,:],Sigma[k,:],UC[k,:],tt,smax)
    T[k,:,:] = tt@Q.T
  return (kappa,Sigma,UC,T)

####
# alternative to CKernelMulti, almost same functionality but has a range of kappa
####

# Inputs:
# A = in-in overlap matrix, shape (n,n)
# mBhalf = in-out overlap matrix, shape (n_out,m,n)
# C = out-out overlap, vector, length n_out
# kappa_array = eigenvalue nodes, vector, length nv, ascending order
# ucmin = minimum U/C (after this focus on noise)
#
# Named inputs:
# smax = maximum noise to allow
#
# Outputs:
# kappa, shape (n_out, m)
# S, shape (n_out, m)
# UC, shape (n_out,m)
# T, shape (n_out,m,n)
#
def get_coadd_matrix_discrete(A, mBhalf, C, kappa_array, ucmin, smax=0.5):

  # get parameters
  (n_out, m, n) = numpy.shape(mBhalf)
  nv = numpy.size(kappa_array)

  # Cholesky decompositions for each eigenvalue node
  L = numpy.zeros((nv,n,n))
  di = numpy.diag_indices(n)
  AA = numpy.copy(A)
  for j in range(nv):
    if j>0:
      AA[di] += kappa_array[j] - kappa_array[j-1]
    else:
      AA[di] += kappa_array[0]
    try:
      L[j,:,:] = scipy.linalg.cholesky(AA, lower=True, check_finite=False)
    except:
      # if matrix is not quite positive definite, we can rectify it
      w, v = numpy.linalg.eigh(A)
      AA[di] += kappa_array[j] + numpy.abs(w[0])
      del v
      #warnings.warn('Warning: pyimcom_lakernel Cholesky decomposition failed; fixed negative eigenvalue {:19.12e}'.format(w[0]))
      L[j,:,:] = scipy.linalg.cholesky(AA, lower=True, check_finite=False)
  del AA; del di

  # outputs
  T_ = numpy.zeros((n_out, m, n), dtype=numpy.float32)
  UC_ = numpy.zeros((n_out, m), dtype=numpy.float32)
  S_ = numpy.zeros((n_out, m), dtype=numpy.float32)
  k_ = numpy.zeros((n_out, m), dtype=numpy.float32)

  Tpi = numpy.zeros((nv, m, n))
  #
  # loop over output PSFs
  for j_out in range(n_out):

    # build values at nodes
    for p in range(nv):
      Tpi[p,:,:] = scipy.linalg.cho_solve((L[p,:,:],True), mBhalf[j_out,:,:].T, check_finite=False).T
    Dp = numpy.einsum('ai,pai->ap', mBhalf[j_out,:,:], Tpi)
    Npq = numpy.einsum('pai,qai->apq', Tpi, Tpi)
    Ep = numpy.einsum('qai,ai->aq', Tpi, mBhalf[j_out,:,:])
    Epq = numpy.zeros((m,nv,nv))
    for p in range(nv):
      for q in range(p):
        Epq[:,q,p] = Epq[:,p,q] = Ep[:,q]  - kappa_array[p]*Npq[:,p,q]
      Epq[:,p,p] = Ep[:,p]  - kappa_array[p]*Npq[:,p,p]

    # now make outputs and call C function
    out_kappa = numpy.zeros((m,))
    out_Sigma = numpy.zeros((m,))
    out_UC = numpy.zeros((m,))
    out_w = numpy.zeros((m*nv,))
    pyimcom_croutines.build_reduced_T_wrap(Npq.flatten(), Dp.flatten()/C[j_out], Epq.flatten()/C[j_out], kappa_array/C[j_out], ucmin, smax,
      out_kappa, out_Sigma, out_UC, out_w)

    # make outputs
    k_[j_out,:] = out_kappa*C[j_out]
    S_[j_out,:] = out_Sigma
    UC_[j_out,:] = out_UC
    T_[j_out,:,:] = numpy.einsum('pai,ap->ai', Tpi, out_w.reshape((m,nv)))

  return (k_, S_, UC_, T_)


# test circuit

def test_get_coadd_matrix_discrete():

  # number of outputs to print
  npr = 4

  # number of layers to test multi-ouptut
  nt = 3

  sigma = 1.75
  u = numpy.array([.2,.1])

  # test grid: interpolate an m1xm1 image from n1xn1
  m1 = 50; n1 = 80
  n = n1*n1
  m = m1*m1

  x = numpy.zeros((n,))
  y = numpy.zeros((n,))
  for i in range(n1):
    y[n1*i:n1*i+n1] = i
    x[i::n1] = i
  xout = numpy.zeros((m,))
  yout = numpy.zeros((m,))
  for i in range(m1):
    yout[m1*i:m1*i+m1] = 5+.25*i
    xout[i::m1] = 5+.25*i

  # make sample image
  thisImage = numpy.exp(2*numpy.pi*1j*(u[0]*x+u[1]*y))
  desiredOutput = numpy.exp(2*numpy.pi*1j*(u[0]*xout+u[1]*yout))

  A = numpy.zeros((n,n))
  mBhalf = numpy.zeros((m,n))
  mBhalfPoly = numpy.zeros((nt,m,n))
  C = numpy.ones((nt,))
  for i in range(n):
    for j in range(n):
      A[i,j] = numpy.exp(-1./sigma**2*( (x[i]-x[j])**2 + (y[i]-y[j])**2 ))
    for a in range(m):
      mBhalf[a,i] = numpy.exp(-1./sigma**2*( (x[i]-xout[a])**2 + (y[i]-yout[a])**2 ))
      for k in range(nt):
        mBhalfPoly[k,a,i] = numpy.exp(-1./sigma**2/(.5+.5*1.05**k)*( (x[i]-xout[a])**2 + (y[i]-yout[a])**2 ))
  for k in range(nt): C[k] = (1+1.05**k)**2/4./1.05**k

  # rescale everything
  A *= .7; mBhalf*= .7; mBhalfPoly *= .7; C *= .7

  fits.PrimaryHDU(A).writeto('A.fits', overwrite=True)
  fits.PrimaryHDU(mBhalfPoly).writeto('mBhalf.fits', overwrite=True)
  print('C=', C)

  kappa_array = numpy.logspace(-6,-2,3)
  print('kappa_array=', kappa_array)

  print('n', n, 'm', m, 'nt', nt, 'nv', numpy.size(kappa_array))

  t1a = time.perf_counter()
  print('begin', t1a)

  (kappa_, Sigma_, UC_, T_) = get_coadd_matrix_discrete(A, mBhalfPoly, C, kappa_array, smax=.5)

  t1b = time.perf_counter()
  print('get_coadd time', t1b-t1a)

  # print information
  fits.PrimaryHDU(T_).writeto('T.fits', overwrite=True)
  fits.PrimaryHDU(UC_).writeto('UC.fits', overwrite=True)
  fits.PrimaryHDU(Sigma_).writeto('Sigma.fits', overwrite=True)
  fits.PrimaryHDU(kappa_).writeto('kappa_ind.fits', overwrite=True)

# this is a test case for the kernel
# nothing fancy. uses to interpolate an image containing a single sine wave, with Gaussian PSF
# inputs:
#   sigma = 1 sigma width of PSF (Gaussian)
#   u (2D numpy array or list) = Fourier wave vector of sine wave. x component first, then y
def testkernel(sigma,u):

  # number of outputs to print
  npr = 4

  # number of layers to test multi-ouptut
  nt = 3

  # test grid: interpolate an m1xm1 image from n1xn1
  m1 = 25; n1 = 33
  n = n1*n1
  m = m1*m1

  x = numpy.zeros((n,))
  y = numpy.zeros((n,))
  for i in range(n1):
    y[n1*i:n1*i+n1] = i
    x[i::n1] = i
  xout = numpy.zeros((m,))
  yout = numpy.zeros((m,))
  for i in range(m1):
    yout[m1*i:m1*i+m1] = 5+.25*i
    xout[i::m1] = 5+.25*i

  # make sample image
  thisImage = numpy.exp(2*numpy.pi*1j*(u[0]*x+u[1]*y))
  desiredOutput = numpy.exp(2*numpy.pi*1j*(u[0]*xout+u[1]*yout))

  print('build matrices', time.perf_counter())

  A = numpy.zeros((n,n))
  mBhalf = numpy.zeros((m,n))
  mBhalfPoly = numpy.zeros((nt,m,n))
  C = 1.
  for i in range(n):
    for j in range(n):
      A[i,j] = numpy.exp(-1./sigma**2*( (x[i]-x[j])**2 + (y[i]-y[j])**2 ))
    for a in range(m):
      mBhalf[a,i] = numpy.exp(-1./sigma**2*( (x[i]-xout[a])**2 + (y[i]-yout[a])**2 ))
      for k in range(nt):
        mBhalfPoly[k,a,i] = numpy.exp(-1./(1.05**k*sigma)**2*( (x[i]-xout[a])**2 + (y[i]-yout[a])**2 ))

  # rescale everything
  A *= .7; mBhalf*= .7; mBhalfPoly *= .7; C *= .7

  t1a = time.perf_counter()
  print('kernel, brute force', t1a)

  # brute force version of kernel
  (kappa,Sigma,UC,T) = BruteForceKernel(A,mBhalf,C,1e-8)

  print('** brute force kernel **')
  print('kappa =', kappa[:npr])
  print('Sigma =', Sigma[:npr])
  print('UC =', UC[:npr])
  print('Image residual =')
  print(numpy.abs(T@thisImage - desiredOutput).reshape((m1,m1))[:npr])

  t1b = time.perf_counter()
  print('kernel, C', t1b)

  # C version of kernel
  (kappa2,Sigma2,UC2,T2) = CKernel(A,mBhalf,C,1e-8)

  print('** C kernel **')
  print('kappa =', kappa2[:npr])
  print('Sigma =', Sigma2[:npr])
  print('UC =', UC2[:npr])
  print('Image residual =')
  print(numpy.abs(T2@thisImage - desiredOutput).reshape((m1,m1))[:npr])

  t1c = time.perf_counter()

  (kappa3,Sigma3,UC3,T3) = CKernelMulti(A,mBhalfPoly,C*1.05**(2*numpy.array(range(nt))),1e-8*numpy.ones((nt,)))
  print('Sigma3 =', Sigma3[:,:npr])
  print('output =', (T2@thisImage)[:npr], (T3@thisImage)[:,:npr])

  t1d = time.perf_counter()
  print('end -->', t1d)

  print('timing: ', t1b-t1a, t1c-t1b, t1d-t1c)

# test interpolation functions
#   u (2D numpy array or list) = Fourier wave vector of sine wave. x component first, then y
def testinterp(u):
  ny = 1024; nx = 1024
  indata = numpy.zeros((3,ny,nx))
  indata[0,:,:] = 1.
  for ix in range(nx):
    indata[1,:,ix] = u[0]*ix + u[1]*numpy.linspace(0,ny-1,ny)
  indata[2,:,:] = numpy.cos(2*numpy.pi*indata[1,:,:])

  no = 32768
  xout = numpy.linspace(8,9,no)
  yout = numpy.linspace(10,10.5,no)

  fout = numpy.zeros((3,no))

  t1a = time.perf_counter()
  pyimcom_croutines.iD5512C(indata, xout, yout, fout)
  #pyimcom_croutines.iD5512C(indata[2,:,:].reshape((1,ny,nx)), xout, yout, fout[2,:].reshape((1,no)))
  t1b = time.perf_counter()

  pred = u[0]*xout + u[1]*yout

  #print(fout)
  #print(pred)
  #print(numpy.cos(2*numpy.pi*pred))
  print('errors:')
  print(fout[0,:]-1)
  print(fout[1,:]-pred)
  print(fout[2,:]-numpy.cos(2*numpy.pi*pred))

  print('timing interp = {:9.6f} s'.format(t1b-t1a))

# tests to run if this is the main function
if __name__ == "__main__":
  testkernel(4., [.2,.1])
