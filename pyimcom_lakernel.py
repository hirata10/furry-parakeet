import numpy
import pyimcom_croutines
import time
import jax

print('......lak4')

# 'Brute force' version of the kernel
# Slow and useful only for comparisons
#
# Inumpyuts:
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
  # eigensystem
  lam, Q = jax.numpy.linalg.eigh(A)
  lam = numpy.asarray(lam)
  Q = numpy.asarray(Q)

  tmp = jax.jit(BruteForceKernel_, static_argnames=['lam','Q','A','mBhalf','C','targetleak','kCmin','kCmax','nbis'])

  return tmp(lam,Q,A,mBhalf,C,targetleak,kCmin=1e-16,kCmax=1e16,nbis=53)

def BruteForceKernel_(lam,Q,A,mBhalf,C,targetleak,kCmin=1e-16,kCmax=1e16,nbis=53):

  # get dimensions and mPhalf matrix
  if mBhalf.ndim==2:
    nt=1
    (m,n) = numpy.shape(mBhalf)
    mBhalf_image = mBhalf.reshape((1,m,n))
    C_s = numpy.asarray([C])
    targetleak_s = numpy.asarray([targetleak])
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

  for k in range(nt):
    # -P/2 matrix
    mPhalf = numpy.matmul(mBhalf[k,:,:],Q)

    # now loop over pixels
    for a in range(m):
      factor = numpy.sqrt(kCmax/kCmin)
      kappa[k,a] = numpy.sqrt(kCmax*kCmin)
      for ibis in range(nbis+1):
        factor = numpy.sqrt(factor)
        UC[k,a] = 1-numpy.sum((lam+2*kappa[k,a])/(lam+kappa[k,a])**2*mPhalf[a,:]**2)/C_s[k]
        if ibis!=nbis:
          if UC[k,a]>targetleak_s[k]:
            kappa[k,a] /= factor
          else:
            kappa[k,a] *= factor
      T[k,a,:] = numpy.matmul(Q,(mPhalf[a,:]/(lam+kappa[k,a])))
      Sigma[k,a] = numpy.sum((mPhalf[a,:]/(lam+kappa[k,a]))**2)

    T[k,:,:] = T[k,:,:]@Q.T

  return (kappa,Sigma,UC,T)

#
# ... and the same function but wrapped around C:
def CKernel(A,mBhalf,C,targetleak,kCmin=1e-16,kCmax=1e16,nbis=53):

  # get dimensions
  (m,n) = numpy.shape(mBhalf)

  # eigensystem
  lam, Q = jax.numpy.linalg.eigh(A)
  lam = numpy.asarray(lam)
  Q = numpy.asarray(Q)
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
# if 2D arrays are inumpyut then assumes nt=1
#
# Inumpyuts:
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
  lam, Q = jax.numpy.linalg.eigh(A)
  lam = numpy.asarray(lam)
  Q = numpy.asarray(Q)

  # get dimensions and mPhalf matrix
  if mBhalf.ndim==2:
    nt=1
    (m,n) = numpy.shape(mBhalf)
    mBhalf_image = mBhalf.reshape((1,m,n))
    C_s = numpy.asarray([C])
    targetleak_s = numpy.asarray([targetleak])
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
    pyimcom_croutines.lakernel1(    lam,
                                    Q,
                                    mBhalf_image[k,:,:]@Q,
                                    C_s[k],
                                    targetleak_s[k],
                                    kCmin,
                                    kCmax,
                                    nbis,
                                    kappa[k,:],
                                    Sigma[k,:],
                                    UC[k,:],
                                    tt,
                                    smax)
    T[k,:,:] = tt@Q.T
  return (kappa,Sigma,UC,T)

# this is a test case for the kernel
# nothing fancy. uses to interpolate an image containing a single sine wave, with Gaussian PSF
# inumpyuts:
#   sigma = 1 sigma width of PSF (Gaussian)
#   u (2D numpy array or list) = Fourier wave vector of sine wave. x component first, then y
def testkernel(sigma,u):

  # number of outputs to print
  numpyr = 4

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
  print('kappa =', kappa[:numpyr])
  print('Sigma =', Sigma[:numpyr])
  print('UC =', UC[:numpyr])
  print('Image residual =')
  print(numpy.abs(T@thisImage - desiredOutput).reshape((m1,m1))[:numpyr])

  t1b = time.perf_counter()
  print('kernel, C', t1b)

  # C version of kernel
  (kappa2,Sigma2,UC2,T2) = CKernel(A,mBhalf,C,1e-8)

  print('** C kernel **')
  print('kappa =', kappa2[:numpyr])
  print('Sigma =', Sigma2[:numpyr])
  print('UC =', UC2[:numpyr])
  print('Image residual =')
  print(numpy.abs(T2@thisImage - desiredOutput).reshape((m1,m1))[:numpyr])

  t1c = time.perf_counter()

  (kappa3,Sigma3,UC3,T3) = CKernelMulti(A,mBhalfPoly,C*1.05**(2*numpy.array(range(nt))),1e-8*numpy.ones((nt,)))
  print('Sigma3 =', Sigma3[:,:numpyr])
  print('output =', (T2@thisImage)[:numpyr], (T3@thisImage)[:,:numpyr])

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
