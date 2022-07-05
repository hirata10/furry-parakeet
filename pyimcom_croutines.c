#include <string.h>
#include <Python.h>
#include "ndarrayobject.h"

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
 * > T = coaddition matrix, shape=(m,n) [needs to be multiplied by Q^T after the return]
 */
static PyObject *pyimcom_lakernel1(PyObject *self, PyObject *args) {

  double C, targetleak, kCmin, kCmax;
  PyObject *lam, *Q, *mPhalfPy; /* inputs */
  PyObject *kappa, *Sigma, *UC, *T; /* outputs */
  PyArrayObject *lam_, *Q_, *mPhalfPy_; /* inputs */
  PyArrayObject *kappa_, *Sigma_, *UC_, *T_; /* outputs */

  long m,n,i,j,a;
  double *mPhalf, *QT_C, *lam_C;
  double *x1, *x2;
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
    sum = 0.;
    for(i=0;i<n;i++) {
      *(double*)PyArray_GETPTR2(T_,a,i)	= var = (*x1++)/((*x2++)+kap);
      sum += var*var;
    }
    *(double*)PyArray_GETPTR1(Sigma_,a) = sum;
    *(double*)PyArray_GETPTR1(kappa_,a) = kap;
    *(double*)PyArray_GETPTR1(UC_,a) = udc;
  }
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

/* 2D, 10x10 kernel interpolation for high accuracy
 * Can interpolate multiple functions at a time from the same grid
 * so we don't have to keep recomputing weights.
 * 
 * Inputs:
 *   infunc = input function on some grid, shape =(nlayer,ngy,ngx)
 *   xpos = input x values, shape=(nout,)
 *   ypos = input y values, shape=(nout,)
 *
 * Outputs:
 * > fhatout = location to put the output values, shape=(nlayer,nout)
 */
static PyObject *pyimcom_iD5512C(PyObject *self, PyObject *args) {

  int i,j,l;
  long nlayer, nout, ngy, ngx, ipos, ilayer;
  double x, y, xfh, yfh; long xi, yi; /* frac and integer parts of abscissae; note 'xfh' and 'yfh' will have 1/2 subtracted */
  double wx[10], wy[10];
  double c,s;
  double interp_vstrip, out;
  long ds;
  char *temp;

  /* Input/output arrays */
  PyObject *infunc, *xpos, *ypos, *fhatout;
  PyArrayObject *infunc_, *xpos_, *ypos_, *fhatout_;

  /* table of values for interpolation formula */
  double zeta[] = {
     7.795042160878816462e-02,  2.269252977160159945e-01,  3.557380180911379752e-01,  4.529461196132943956e-01,  5.099362658787808256e-01
  };
  double HmatrixC[] = {
     1.912402678501005084e+03, -4.927004100469148398e+03,  5.835905613163729868e+03, -4.322722449499965478e+03,  1.501418877063505988e+03,
    -1.217699386087176026e+04,  3.159481253500127059e+04, -3.785475949046354799e+04,  2.836995380606032268e+04, -9.933019743019915040e+03,
     3.246330126827143977e+04, -8.462064453664862958e+04,  1.021891619223469461e+05, -7.724212924975770875e+04,  2.721034680139671400e+04,
    -4.342904595880888519e+04,  1.135281945624359505e+05, -1.377799817082234076e+05,  1.047254797516023391e+05, -3.704478343160567601e+04,
     2.123096597676863894e+04, -5.557555073910982173e+04,  6.760976692036786699e+04, -5.153062474798672338e+04,  1.826604930348573544e+04,
     2.123096597676863894e+04, -5.557555073910982173e+04,  6.760976692036786699e+04, -5.153062474798672338e+04,  1.826604930348573544e+04,
    -4.342904595880888519e+04,  1.135281945624359505e+05, -1.377799817082234076e+05,  1.047254797516023391e+05, -3.704478343160567601e+04,
     3.246330126827143977e+04, -8.462064453664862958e+04,  1.021891619223469461e+05, -7.724212924975770875e+04,  2.721034680139671400e+04,
    -1.217699386087176026e+04,  3.159481253500127059e+04, -3.785475949046354799e+04,  2.836995380606032268e+04, -9.933019743019915040e+03,
     1.912402678501005084e+03, -4.927004100469148398e+03,  5.835905613163729868e+03, -4.322722449499965478e+03,  1.501418877063505988e+03
  };
  double HmatrixS[] = {
    -4.904230619110763655e+04,  4.323751412374444772e+04, -3.246339075532347488e+04,  1.875968952461114532e+04, -5.760491925503920356e+03,
     4.103555938606222626e+05, -3.637390867586739478e+05,  2.755014430530755781e+05, -1.606389018624043674e+05,  4.963098826150417153e+04,
    -1.555126539182490204e+06,  1.383601738371265586e+06, -1.054523731121562887e+06,  6.189727207369643729e+05, -1.921388961754927877e+05,
     3.501335761714349966e+06, -3.122480589327111840e+06,  2.389400028862954117e+06, -1.408673167242457159e+06,  4.386664738897074712e+05,
    -5.159499149133198895e+06,  4.606470743687568232e+06, -3.531921534316427074e+06,  2.086791191489480436e+06, -6.508774765937846387e+05,
     5.159499149133198895e+06, -4.606470743687568232e+06,  3.531921534316427074e+06, -2.086791191489480436e+06,  6.508774765937846387e+05,
    -3.501335761714349966e+06,  3.122480589327111840e+06, -2.389400028862954117e+06,  1.408673167242457159e+06, -4.386664738897074712e+05,
     1.555126539182490204e+06, -1.383601738371265586e+06,  1.054523731121562887e+06, -6.189727207369643729e+05,  1.921388961754927877e+05,
    -4.103555938606222626e+05,  3.637390867586739478e+05, -2.755014430530755781e+05,  1.606389018624043674e+05, -4.963098826150417153e+04,
     4.904230619110763655e+04, -4.323751412374444772e+04,  3.246339075532347488e+04, -1.875968952461114532e+04,  5.760491925503920356e+03,
  };

  /* read arguments */
  if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &infunc, &PyArray_Type, &xpos, &PyArray_Type, &ypos,
    &PyArray_Type, &fhatout)) {

    return(NULL);
  }
  /* ... and repackage into C objects */
  infunc_ = (PyArrayObject*)PyArray_FROM_OTF(infunc, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  xpos_ = (PyArrayObject*)PyArray_FROM_OTF(xpos, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  ypos_ = (PyArrayObject*)PyArray_FROM_OTF(ypos, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  fhatout_ = (PyArrayObject*)PyArray_FROM_OTF(fhatout, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

  /* extract dimensions */
  nlayer = infunc_->dimensions[0];
  ngy = infunc_->dimensions[1];
  ngx = infunc_->dimensions[2];
  nout = xpos_->dimensions[0];
  /* .. and strides .. */
  ds = infunc_->strides[2];

  /* loop over points to interpolate */
  for(ipos=0;ipos<nout;ipos++) {
    x = *(double*)PyArray_GETPTR1(xpos_,ipos);
    y = *(double*)PyArray_GETPTR1(ypos_,ipos);
    xi = (long)floor(x);
    yi = (long)floor(y);
    xfh = x-xi-.5;
    yfh = y-yi-.5;
    if (xi<4 || xi>=ngx-5 || yi<4 || yi>=ngy-5) continue; /* point off the grid, don't interpolate */

    /* now compute the weights */
    for(j=0;j<10;j++) wx[j] = wy[j] = 0.;
    for(l=0;l<5;l++) {
      c = cos(zeta[l]*xfh);
      s = sin(zeta[l]*xfh);
      for(j=0;j<10;j++) wx[j] += HmatrixC[5*j+l]*c + HmatrixS[5*j+l]*s;
      c = cos(zeta[l]*yfh);
      s = sin(zeta[l]*yfh);
      for(j=0;j<10;j++) wy[j] += HmatrixC[5*j+l]*c + HmatrixS[5*j+l]*s;
    }

#if 0
    /* testing only */
    printf("pos %3ld: %3ld %3ld\n", ipos, xi, yi);
    for(j=0;j<10;j++) printf("  %11.8lf\n", wx[j]); printf("\n");
    for(j=0;j<10;j++) printf("  %11.8lf\n", wy[j]); printf("\n");
    printf("  <%12.5lE>\n", *(double*)PyArray_GETPTR3(infunc_,1,yi-4,xi-4));
#endif

    /* and the outputs */
    for(ilayer=0;ilayer<nlayer;ilayer++) {
      out = 0.;
      for(i=0;i<10;i++) {
        interp_vstrip = 0.;
        temp = (char*)PyArray_GETPTR3(infunc_,ilayer,yi-4+i,xi-4);
        for(j=0;j<10;j++) {
          interp_vstrip += wx[j]*(*(double*)temp);
          temp += ds;
        }
        out += interp_vstrip*wy[i];
      }
      *(double*)PyArray_GETPTR2(fhatout_, ilayer, ipos) = out;
    }
  }


  Py_DECREF(infunc_);
  Py_DECREF(xpos_);
  Py_DECREF(ypos_);
  PyArray_ResolveWritebackIfCopy(fhatout_);
  Py_DECREF(fhatout_);

  return(Py_None);
  /* -- this is the end of the function if it executed normally -- */
}

/* Method Table */
static PyMethodDef PyImcom_CMethods[] = {
  {"lakernel1", (PyCFunction)pyimcom_lakernel1, METH_VARARGS, "PyIMCOM core linear algebra kernel"},
  {"iD5512C", (PyCFunction)pyimcom_iD5512C, METH_VARARGS, "interpolation routine"},
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

