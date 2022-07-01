import numpy
import pyimcom_croutines
import time

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
  T = numpy.zeros((m,n))

  pyimcom_croutines.lakernel1(lam,Q,mPhalf,C,targetleak,kCmin,kCmax,nbis,kappa,Sigma,UC,T)
  return (kappa,Sigma,UC,T)

# this is a test case for the kernel
# nothing fancy. uses to interpolate an image containing a single sine wave, with Gaussian PSF
# inputs:
#   sigma = 1 sigma width of PSF (Gaussian)
#   u (2D numpy array or list) = Fourier wave vector of sine wave. x component first, then y
def testkernel(sigma,u):

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
  C = 1.
  for i in range(n):
    for j in range(n):
      A[i,j] = numpy.exp(-1./sigma**2*( (x[i]-x[j])**2 + (y[i]-y[j])**2 ))
    for a in range(m):
      mBhalf[a,i] = numpy.exp(-1./sigma**2*( (x[i]-xout[a])**2 + (y[i]-yout[a])**2 ))

  t1a = time.perf_counter()
  print('kernel, brute force', t1a)

  # brute force version of kernel
  (kappa,Sigma,UC,T) = BruteForceKernel(A,mBhalf,C,1e-8)

  print('** brute force kernel **')
  print('kappa =', kappa)
  print('Sigma =', Sigma)
  print('UC =', UC)
  print('Image residual =')
  print(numpy.abs(T@thisImage - desiredOutput).reshape((m1,m1)))

  t1b = time.perf_counter()
  print('kernel, C', t1b)

  # C version of kernel
  (kappa2,Sigma2,UC2,T2) = CKernel(A,mBhalf,C,1e-8)

  print('** C kernel **')
  print('kappa =', kappa2)
  print('Sigma =', Sigma2)
  print('UC =', UC2)
  print('Image residual =')
  print(numpy.abs(T2@thisImage - desiredOutput).reshape((m1,m1)))

  t1c = time.perf_counter()
  print('end -->', t1c)

  print('timing: ', t1b-t1a, t1c-t1b)

# tests to run if this is the main function
if __name__ == "__main__":
  testkernel(4., [.2,.1])
