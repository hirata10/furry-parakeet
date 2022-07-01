#include <string.h>
#include <Python.h>
#include "ndarrayobject.h"

#define IS_TIMING

#ifdef IS_TIMING
#include <stdio.h>
#include <time.h>
#endif

/* PyIMCOM linear algebra kernel -- all the steps with the for loops
 * (i.e., except the matrix diagonalization)
 *
 * Inputs:
 *   lam = system matrix eigenvalues, shape=(n,)
 *   Q = system matrix eigenvectors, shape=(n,n)
 *   mPhalf = -P/2 = premultiplied target overlap matrix, shape=(m,n)
 *   C = target normalization (scalar)
 *   targetleak = allowable leakage of target PSF
 *   kCmin, kCmax, nbis = range of kappa/C to test, number of bisections
 *
 * Outputs:
 * > kappa = Lagrange multiplier per output pixel, shape=(m,)
 * > Sigma = output noise amplification, shape=(m,)
 * > UC = fractional squared error in PSF, shape=(m,)
 * > T = coaddition matrix, shape=(m,n)
 */
static PyObject *pyimcom_lakernel1(PyObject *self, PyObject *args) {

  double C, targetleak, kCmin, kCmax;
  PyObject *lam, *Q, *mPhalfPy; /* inputs */
  PyObject *kappa, *Sigma, *UC, *T; /* outputs */
  PyArrayObject *lam_, *Q_, *mPhalfPy_; /* inputs */
  PyArrayObject *kappa_, *Sigma_, *UC_, *T_; /* outputs */

  long m,n,i,j,a;
  double *mPhalf, *QT_C, *lam_C;
  double *x1, *x2, *x3, *x4, *x5;
  double factor, kap, dkap, udc, sum, var;
  int ib, nbis;

#ifdef IS_TIMING
  clock_t t1;

  t1 = clock();
#endif

  /* read arguments */
  if (!PyArg_ParseTuple(args, "O!O!O!ddddiO!O!O!O!", &PyArray_Type, &lam, &PyArray_Type, &Q, &PyArray_Type, &mPhalfPy,
    &C, &targetleak, &kCmin, &kCmax, &nbis,
    &PyArray_Type, &kappa, &PyArray_Type, &Sigma, &PyArray_Type, &UC, &PyArray_Type, &T)) {

    return(NULL);
  }

  /* repackage Python arrays as C objects */
  lam_ = (PyArrayObject*)PyArray_FROM_OTF(lam, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  Q_ = (PyArrayObject*)PyArray_FROM_OTF(Q, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  mPhalfPy_ = (PyArrayObject*)PyArray_FROM_OTF(mPhalfPy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  kappa_ = (PyArrayObject*)PyArray_FROM_OTF(kappa, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
  Sigma_ = (PyArrayObject*)PyArray_FROM_OTF(Sigma, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
  UC_ = (PyArrayObject*)PyArray_FROM_OTF(UC, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
  T_ = (PyArrayObject*)PyArray_FROM_OTF(T, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

#ifdef IS_TIMING
  printf("Time 1, %9.6lf\n", (clock()-t1)/(double)CLOCKS_PER_SEC);
#endif
  /* -- unpackaging complete -- */

  /* dimensions */
  m = mPhalfPy_->dimensions[0];
  n = mPhalfPy_->dimensions[1];

  /* package inputs into flat C arrays to minimize pointer arithmetic in inner for loops */
  mPhalf = (double*)malloc((size_t)(m*n*sizeof(double)));
  for(a=0;a<m;a++) for(j=0;j<n;j++) mPhalf[a*n+j] = *(double*)PyArray_GETPTR2(mPhalfPy_,a,j);
  /* .. and this one is Q^T */
  QT_C = (double*)malloc((size_t)(n*n*sizeof(double)));
  for(i=0;i<n;i++) for(j=0;j<n;j++) QT_C[j*n+i] = *(double*)PyArray_GETPTR2(Q_,i,j);
  lam_C = (double*)malloc((size_t)(n*sizeof(double)));
  for(i=0;i<n;i++) lam_C[i] = *(double*)PyArray_GETPTR1(lam_,i);
#ifdef IS_TIMING
  printf("Time 2, %9.6lf\n", (clock()-t1)/(double)CLOCKS_PER_SEC);
#endif

#ifdef IS_TIMING
  printf("Time 3, %9.6lf\n", (clock()-t1)/(double)CLOCKS_PER_SEC);
#endif

  /* now loop over pixels */
  x4 = (double*)malloc((size_t)(n*sizeof(double)));
  for(a=0;a<m;a++) {
    factor = sqrt(kCmax/kCmin);
    kap = sqrt(kCmax*kCmin);
    for(ib=0;ib<=nbis;ib++) {
      dkap = 2.*kap;
      sum = 0.;
      x1 = mPhalf+a*n;
      x2 = lam_C;
      for(i=0;i<n;i++) {
        var = (*x1++)/(*x2+kap);
        sum += ((*x2++) + dkap)*var*var;
      }
      udc = 1.-sum/C;
      if(ib!=nbis) {
        kap *= udc>targetleak? 1./factor: factor;
        factor=sqrt(factor);
      }
    }

    /* report T and Sigma */
    x1 = mPhalf+a*n;
    x2 = lam_C;
    x3 = QT_C;
    sum = 0.;
    memset(x4, 0, n*sizeof(double));
    for(i=0;i<n;i++) {
      var = (*x1++)/((*x2++)+kap);
      sum += var*var;
      x5=x4;
      for(j=0;j<n;j++) *x5++ += (*x3++)*var;
    }
    for(j=0;j<n;j++) *(double*)PyArray_GETPTR2(T_,a,j) = x4[j];
    *(double*)PyArray_GETPTR1(Sigma_,a) = sum;
    *(double*)PyArray_GETPTR1(kappa_,a) = kap;
    *(double*)PyArray_GETPTR1(UC_,a) = udc;

#ifdef IS_TIMING
  if (a==0) printf("Time 3.5C, %9.6lf\n", (clock()-t1)/(double)CLOCKS_PER_SEC);
#endif
  }
  free((char*)x4);
#ifdef IS_TIMING
  printf("Time 4, %9.6lf\n", (clock()-t1)/(double)CLOCKS_PER_SEC);
#endif

  /* reference count and resolve */
  Py_DECREF(lam_);
  Py_DECREF(Q_);
  Py_DECREF(mPhalfPy_);
  PyArray_ResolveWritebackIfCopy(kappa_);
  Py_DECREF(kappa_);
  PyArray_ResolveWritebackIfCopy(Sigma_);
  Py_DECREF(Sigma_);
  PyArray_ResolveWritebackIfCopy(UC_);
  Py_DECREF(UC_);
  PyArray_ResolveWritebackIfCopy(T_);
  Py_DECREF(T_);

  /* cleanup memory */
  free((char*)QT_C);
  free((char*)lam_C);
  free((char*)mPhalf);

  Py_INCREF(Py_None);
#ifdef IS_TIMING
  printf("Time End, %9.6lf\n", (clock()-t1)/(double)CLOCKS_PER_SEC);
#endif
  return(Py_None);
  /* -- this is the end of the function if it executed normally -- */

}

/* Method Table */
static PyMethodDef PyImcom_CMethods[] = {
  {"lakernel1", pyimcom_lakernel1, METH_VARARGS, "PyIMCOM core linear algebra kernel"},
  /* more functions, if needed */
  {NULL, NULL, 0, NULL} /* end */
};

/* Module Definition Structure */
static struct PyModuleDef pyimcom_croutinesmodule = {
  PyModuleDef_HEAD_INIT,
  "pyimcom_croutines",
  NULL,
  -1,
  PyImcom_CMethods
};

PyMODINIT_FUNC PyInit_pyimcom_croutines(void) {
  import_array(); /* needed for numpy arrays */
  return PyModule_Create(&pyimcom_croutinesmodule);
}

