#include <string.h>
#include <Python.h>
#include "ndarrayobject.h"

#ifdef IS_TIMING
#include <stdio.h>
#include <stdlib.h>
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
 *   smax = maximum allowed Sigma
 *
 * Outputs:
 * > kappa = Lagrange multiplier per output pixel, shape=(m,)
 * > Sigma = output noise amplification, shape=(m,)
 * > UC = fractional squared error in PSF, shape=(m,)
 * > T = coaddition matrix, shape=(m,n) [needs to be multiplied by Q^T after the return]
 */
static PyObject *pyimcom_lakernel1(PyObject *self, PyObject *args) {

  double C, targetleak, kCmin, kCmax, smax;
  PyObject *lam, *Q, *mPhalfPy; /* inputs */
  PyObject *kappa, *Sigma, *UC, *T; /* outputs */
  PyArrayObject *lam_, *Q_, *mPhalfPy_; /* inputs */
  PyArrayObject *kappa_, *Sigma_, *UC_, *T_; /* outputs */

  long m,n,i,j,a;
  double *mPhalf, *QT_C, *lam_C;
  double *x1, *x2;
  double factor, kap, dkap, udc=1., sum, sum2, var;
  int ib, nbis;

#ifdef IS_TIMING
  clock_t t1;

  t1 = clock();
#endif

  /* read arguments */
  if (!PyArg_ParseTuple(args, "O!O!O!ddddiO!O!O!O!d", &PyArray_Type, &lam, &PyArray_Type, &Q, &PyArray_Type, &mPhalfPy,
    &C, &targetleak, &kCmin, &kCmax, &nbis,
    &PyArray_Type, &kappa, &PyArray_Type, &Sigma, &PyArray_Type, &UC, &PyArray_Type, &T, &smax)) {

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
  mPhalf = (double*)malloc((size_t)((m*n+n*n+n)*sizeof(double)));
  if (mPhalf==NULL) return(NULL);
  for(a=0;a<m;a++) for(j=0;j<n;j++) mPhalf[a*n+j] = *(double*)PyArray_GETPTR2(mPhalfPy_,a,j);
  /* .. and this one is Q^T */
  QT_C = mPhalf + m*n;
  for(i=0;i<n;i++) for(j=0;j<n;j++) QT_C[j*n+i] = *(double*)PyArray_GETPTR2(Q_,i,j);
  lam_C = QT_C+n;
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
      sum = sum2 = 0.;
      x1 = mPhalf+a*n;
      x2 = lam_C;
      for(i=0;i<n;i++) {
        var = (*x1++)/(*x2+kap);
        sum2 += var*var;
        sum += ((*x2++) + dkap)*var*var;
      }
      udc = 1.-sum/C;
      if(ib!=nbis) {
        factor=sqrt(factor);
        kap *= udc>targetleak && sum2<smax? 1./factor: factor;
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

  int i,j;
  long nlayer, nout, ngy, ngx, ipos, ilayer;
  double x, y, xfh, yfh; long xi, yi; /* frac and integer parts of abscissae; note 'xfh' and 'yfh' will have 1/2 subtracted */
  double wx[10], wy[10];
  double e_,o_,xfh2,yfh2;
  double interp_vstrip, out;
  double *locdata, *L2;

  /* Input/output arrays */
  PyObject *infunc, *xpos, *ypos, *fhatout;
  PyArrayObject *infunc_, *xpos_, *ypos_, *fhatout_;

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

  /* get local data */
  locdata = (double*)malloc((size_t)(nlayer*ngy*ngx*sizeof(double)));
  ipos = 0;
  for(ilayer=0;ilayer<nlayer;ilayer++)
    for(yi=0;yi<ngy;yi++)
      for(xi=0;xi<ngx;xi++)
        locdata[ipos++] = *(double*)PyArray_GETPTR3(infunc_,ilayer,yi,xi);

  /* loop over points to interpolate */
  for(ipos=0;ipos<nout;ipos++) {
    x = *(double*)PyArray_GETPTR1(xpos_,ipos);
    y = *(double*)PyArray_GETPTR1(ypos_,ipos);
    xi = (long)floor(x);
    yi = (long)floor(y);
    xfh = x-xi-.5;
    yfh = y-yi-.5;
    if (xi<4 || xi>=ngx-5 || yi<4 || yi>=ngy-5) continue; /* point off the grid, don't interpolate */

    /* -- begin interpolation code written by python --*/
    xfh2 = xfh*xfh;
    e_ =  ((( 1.651881673372979740E-05*xfh2-3.145538007199505447E-04)*xfh2+1.793518183780194427E-03)*xfh2-2.904014557029917318E-03)*xfh2+6.187591260980151433E-04;
    o_ = ((((-3.486978652054735998E-06*xfh2+6.753750285320532433E-05)*xfh2-3.871378836550175566E-04)*xfh2+6.279918076641771273E-04)*xfh2-1.338434614116611838E-04)*xfh;
    wx[0] = e_ + o_;
    wx[9] = e_ - o_;
    e_ =  (((-1.146756217210629335E-04*xfh2+2.883845374976550142E-03)*xfh2-1.857047531896089884E-02)*xfh2+3.147734488597204311E-02)*xfh2-6.753293626461192439E-03;
    o_ = (((( 3.121412120355294799E-05*xfh2-8.040343683015897672E-04)*xfh2+5.209574765466357636E-03)*xfh2-8.847326408846412429E-03)*xfh2+1.898674086370833597E-03)*xfh;
    wx[1] = e_ + o_;
    wx[8] = e_ - o_;
    e_ =  ((( 3.256838096371517067E-04*xfh2-9.702063770653997568E-03)*xfh2+8.678848026470635524E-02)*xfh2-1.659182651092198924E-01)*xfh2+3.620560878249733799E-02;
    o_ = ((((-1.243658986204533102E-04*xfh2+3.804930695189636097E-03)*xfh2-3.434861846914529643E-02)*xfh2+6.581033749134083954E-02)*xfh2-1.436476114189205733E-02)*xfh;
    wx[2] = e_ + o_;
    wx[7] = e_ - o_;
    e_ =  (((-4.541830837949564726E-04*xfh2+1.494862093737218955E-02)*xfh2-1.668775957435094937E-01)*xfh2+5.879306056792649171E-01)*xfh2-1.367845996704077915E-01;
    o_ = (((( 2.894406669584551734E-04*xfh2-9.794291009695265532E-03)*xfh2+1.104231510875857830E-01)*xfh2-3.906954914039130755E-01)*xfh2+9.092432925988773451E-02)*xfh;
    wx[3] = e_ + o_;
    wx[6] = e_ - o_;
    e_ =  ((( 2.266560930061513573E-04*xfh2-7.815848920941316502E-03)*xfh2+9.686607348538181506E-02)*xfh2-4.505856722239036105E-01)*xfh2+6.067135256905490381E-01;
    o_ = ((((-4.336085507644610966E-04*xfh2+1.537862263741893339E-02)*xfh2-1.925091434770601628E-01)*xfh2+8.993141455798455697E-01)*xfh2-1.213035309579723942E+00)*xfh;
    wx[4] = e_ + o_;
    wx[5] = e_ - o_;
    yfh2 = yfh*yfh;
    e_ =  ((( 1.651881673372979740E-05*yfh2-3.145538007199505447E-04)*yfh2+1.793518183780194427E-03)*yfh2-2.904014557029917318E-03)*yfh2+6.187591260980151433E-04;
    o_ = ((((-3.486978652054735998E-06*yfh2+6.753750285320532433E-05)*yfh2-3.871378836550175566E-04)*yfh2+6.279918076641771273E-04)*yfh2-1.338434614116611838E-04)*yfh;
    wy[0] = e_ + o_;
    wy[9] = e_ - o_;
    e_ =  (((-1.146756217210629335E-04*yfh2+2.883845374976550142E-03)*yfh2-1.857047531896089884E-02)*yfh2+3.147734488597204311E-02)*yfh2-6.753293626461192439E-03;
    o_ = (((( 3.121412120355294799E-05*yfh2-8.040343683015897672E-04)*yfh2+5.209574765466357636E-03)*yfh2-8.847326408846412429E-03)*yfh2+1.898674086370833597E-03)*yfh;
    wy[1] = e_ + o_;
    wy[8] = e_ - o_;
    e_ =  ((( 3.256838096371517067E-04*yfh2-9.702063770653997568E-03)*yfh2+8.678848026470635524E-02)*yfh2-1.659182651092198924E-01)*yfh2+3.620560878249733799E-02;
    o_ = ((((-1.243658986204533102E-04*yfh2+3.804930695189636097E-03)*yfh2-3.434861846914529643E-02)*yfh2+6.581033749134083954E-02)*yfh2-1.436476114189205733E-02)*yfh;
    wy[2] = e_ + o_;
    wy[7] = e_ - o_;
    e_ =  (((-4.541830837949564726E-04*yfh2+1.494862093737218955E-02)*yfh2-1.668775957435094937E-01)*yfh2+5.879306056792649171E-01)*yfh2-1.367845996704077915E-01;
    o_ = (((( 2.894406669584551734E-04*yfh2-9.794291009695265532E-03)*yfh2+1.104231510875857830E-01)*yfh2-3.906954914039130755E-01)*yfh2+9.092432925988773451E-02)*yfh;
    wy[3] = e_ + o_;
    wy[6] = e_ - o_;
    e_ =  ((( 2.266560930061513573E-04*yfh2-7.815848920941316502E-03)*yfh2+9.686607348538181506E-02)*yfh2-4.505856722239036105E-01)*yfh2+6.067135256905490381E-01;
    o_ = ((((-4.336085507644610966E-04*yfh2+1.537862263741893339E-02)*yfh2-1.925091434770601628E-01)*yfh2+8.993141455798455697E-01)*yfh2-1.213035309579723942E+00)*yfh;
    wy[4] = e_ + o_;
    wy[5] = e_ - o_;
    /* -- end interpolation code written by python --*/

    /* and the outputs */
    for(ilayer=0;ilayer<nlayer;ilayer++) {
      out = 0.;
      L2 = locdata + (ngy*ilayer + yi-4)*ngx + xi-4;
      for(i=0;i<10;i++) {
        interp_vstrip = 0.;
        for(j=0;j<10;j++) interp_vstrip += wx[j]*L2[j];
        out += interp_vstrip*wy[i];
        L2 += ngx; /* jump to next row of input image */
      }
      *(double*)PyArray_GETPTR2(fhatout_, ilayer, ipos) = out;
    }
  }

  Py_DECREF(infunc_);
  Py_DECREF(xpos_);
  Py_DECREF(ypos_);
  PyArray_ResolveWritebackIfCopy(fhatout_);
  Py_DECREF(fhatout_);

  free((char*)locdata);

  Py_INCREF(Py_None);
  return(Py_None);
  /* -- this is the end of the function if it executed normally -- */
}

/* 2D, 10x10 kernel interpolation for high accuracy
 * Can interpolate multiple functions at a time from the same grid
 * so we don't have to keep recomputing weights.
 *
 * This version assumes the output is symmetrical as a sqrt nout x sqrt nout matrix
 * 
 * Inputs:
 *   infunc = input function on some grid, shape =(nlayer,ngy,ngx)
 *   xpos = input x values, shape=(nout,)
 *   ypos = input y values, shape=(nout,)
 *
 * Outputs:
 * > fhatout = location to put the output values, shape=(nlayer,nout)
 */
static PyObject *pyimcom_iD5512C_sym(PyObject *self, PyObject *args) {

  int i,j;
  long nlayer, nout, ngy, ngx, ipos, ilayer;
  double x, y, xfh, yfh; long xi, yi; /* frac and integer parts of abscissae; note 'xfh' and 'yfh' will have 1/2 subtracted */
  double wx[10], wy[10];
  double e_,o_,xfh2,yfh2;
  double interp_vstrip, out;
  double *locdata, *L2;
  long sqnout, ipos1, ipos2, ipos_sym;

  /* Input/output arrays */
  PyObject *infunc, *xpos, *ypos, *fhatout;
  PyArrayObject *infunc_, *xpos_, *ypos_, *fhatout_;

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
  sqnout = (long)floor(sqrt(nout+1));

  /* get local data */
  locdata = (double*)malloc((size_t)(nlayer*ngy*ngx*sizeof(double)));
  ipos = 0;
  for(ilayer=0;ilayer<nlayer;ilayer++)
    for(yi=0;yi<ngy;yi++)
      for(xi=0;xi<ngx;xi++)
        locdata[ipos++] = *(double*)PyArray_GETPTR3(infunc_,ilayer,yi,xi);

  /* loop over points to interpolate, but in this function only the upper half triangle */
  for(ipos1=0;ipos1<sqnout;ipos1++) for(ipos2=ipos1;ipos2<sqnout;ipos2++) {
    ipos = ipos1*sqnout+ipos2;
    x = *(double*)PyArray_GETPTR1(xpos_,ipos);
    y = *(double*)PyArray_GETPTR1(ypos_,ipos);
    xi = (long)floor(x);
    yi = (long)floor(y);
    xfh = x-xi-.5;
    yfh = y-yi-.5;
    if (xi<4 || xi>=ngx-5 || yi<4 || yi>=ngy-5) continue; /* point off the grid, don't interpolate */

    /* -- begin interpolation code written by python --*/
    xfh2 = xfh*xfh;
    e_ =  ((( 1.651881673372979740E-05*xfh2-3.145538007199505447E-04)*xfh2+1.793518183780194427E-03)*xfh2-2.904014557029917318E-03)*xfh2+6.187591260980151433E-04;
    o_ = ((((-3.486978652054735998E-06*xfh2+6.753750285320532433E-05)*xfh2-3.871378836550175566E-04)*xfh2+6.279918076641771273E-04)*xfh2-1.338434614116611838E-04)*xfh;
    wx[0] = e_ + o_;
    wx[9] = e_ - o_;
    e_ =  (((-1.146756217210629335E-04*xfh2+2.883845374976550142E-03)*xfh2-1.857047531896089884E-02)*xfh2+3.147734488597204311E-02)*xfh2-6.753293626461192439E-03;
    o_ = (((( 3.121412120355294799E-05*xfh2-8.040343683015897672E-04)*xfh2+5.209574765466357636E-03)*xfh2-8.847326408846412429E-03)*xfh2+1.898674086370833597E-03)*xfh;
    wx[1] = e_ + o_;
    wx[8] = e_ - o_;
    e_ =  ((( 3.256838096371517067E-04*xfh2-9.702063770653997568E-03)*xfh2+8.678848026470635524E-02)*xfh2-1.659182651092198924E-01)*xfh2+3.620560878249733799E-02;
    o_ = ((((-1.243658986204533102E-04*xfh2+3.804930695189636097E-03)*xfh2-3.434861846914529643E-02)*xfh2+6.581033749134083954E-02)*xfh2-1.436476114189205733E-02)*xfh;
    wx[2] = e_ + o_;
    wx[7] = e_ - o_;
    e_ =  (((-4.541830837949564726E-04*xfh2+1.494862093737218955E-02)*xfh2-1.668775957435094937E-01)*xfh2+5.879306056792649171E-01)*xfh2-1.367845996704077915E-01;
    o_ = (((( 2.894406669584551734E-04*xfh2-9.794291009695265532E-03)*xfh2+1.104231510875857830E-01)*xfh2-3.906954914039130755E-01)*xfh2+9.092432925988773451E-02)*xfh;
    wx[3] = e_ + o_;
    wx[6] = e_ - o_;
    e_ =  ((( 2.266560930061513573E-04*xfh2-7.815848920941316502E-03)*xfh2+9.686607348538181506E-02)*xfh2-4.505856722239036105E-01)*xfh2+6.067135256905490381E-01;
    o_ = ((((-4.336085507644610966E-04*xfh2+1.537862263741893339E-02)*xfh2-1.925091434770601628E-01)*xfh2+8.993141455798455697E-01)*xfh2-1.213035309579723942E+00)*xfh;
    wx[4] = e_ + o_;
    wx[5] = e_ - o_;
    yfh2 = yfh*yfh;
    e_ =  ((( 1.651881673372979740E-05*yfh2-3.145538007199505447E-04)*yfh2+1.793518183780194427E-03)*yfh2-2.904014557029917318E-03)*yfh2+6.187591260980151433E-04;
    o_ = ((((-3.486978652054735998E-06*yfh2+6.753750285320532433E-05)*yfh2-3.871378836550175566E-04)*yfh2+6.279918076641771273E-04)*yfh2-1.338434614116611838E-04)*yfh;
    wy[0] = e_ + o_;
    wy[9] = e_ - o_;
    e_ =  (((-1.146756217210629335E-04*yfh2+2.883845374976550142E-03)*yfh2-1.857047531896089884E-02)*yfh2+3.147734488597204311E-02)*yfh2-6.753293626461192439E-03;
    o_ = (((( 3.121412120355294799E-05*yfh2-8.040343683015897672E-04)*yfh2+5.209574765466357636E-03)*yfh2-8.847326408846412429E-03)*yfh2+1.898674086370833597E-03)*yfh;
    wy[1] = e_ + o_;
    wy[8] = e_ - o_;
    e_ =  ((( 3.256838096371517067E-04*yfh2-9.702063770653997568E-03)*yfh2+8.678848026470635524E-02)*yfh2-1.659182651092198924E-01)*yfh2+3.620560878249733799E-02;
    o_ = ((((-1.243658986204533102E-04*yfh2+3.804930695189636097E-03)*yfh2-3.434861846914529643E-02)*yfh2+6.581033749134083954E-02)*yfh2-1.436476114189205733E-02)*yfh;
    wy[2] = e_ + o_;
    wy[7] = e_ - o_;
    e_ =  (((-4.541830837949564726E-04*yfh2+1.494862093737218955E-02)*yfh2-1.668775957435094937E-01)*yfh2+5.879306056792649171E-01)*yfh2-1.367845996704077915E-01;
    o_ = (((( 2.894406669584551734E-04*yfh2-9.794291009695265532E-03)*yfh2+1.104231510875857830E-01)*yfh2-3.906954914039130755E-01)*yfh2+9.092432925988773451E-02)*yfh;
    wy[3] = e_ + o_;
    wy[6] = e_ - o_;
    e_ =  ((( 2.266560930061513573E-04*yfh2-7.815848920941316502E-03)*yfh2+9.686607348538181506E-02)*yfh2-4.505856722239036105E-01)*yfh2+6.067135256905490381E-01;
    o_ = ((((-4.336085507644610966E-04*yfh2+1.537862263741893339E-02)*yfh2-1.925091434770601628E-01)*yfh2+8.993141455798455697E-01)*yfh2-1.213035309579723942E+00)*yfh;
    wy[4] = e_ + o_;
    wy[5] = e_ - o_;
    /* -- end interpolation code written by python --*/

    /* and the outputs */
    for(ilayer=0;ilayer<nlayer;ilayer++) {
      out = 0.;
      L2 = locdata + (ngy*ilayer + yi-4)*ngx + xi-4;
      for(i=0;i<10;i++) {
        interp_vstrip = 0.;
        for(j=0;j<10;j++) interp_vstrip += wx[j]*L2[j];
        out += interp_vstrip*wy[i];
        L2 += ngx; /* jump to next row of input image */
      }
      *(double*)PyArray_GETPTR2(fhatout_, ilayer, ipos) = out;
    }
  }

  /* ... and now fill in the lower half triangle */
  for(ipos1=1;ipos1<sqnout;ipos1++) for(ipos2=0;ipos2<ipos1;ipos2++) {
    ipos = ipos1*sqnout+ipos2;
    ipos_sym = ipos2*sqnout+ipos1;
    for(ilayer=0;ilayer<nlayer;ilayer++)
      *(double*)PyArray_GETPTR2(fhatout_, ilayer, ipos) = *(double*)PyArray_GETPTR2(fhatout_, ilayer, ipos_sym);
  }

  Py_DECREF(infunc_);
  Py_DECREF(xpos_);
  Py_DECREF(ypos_);
  PyArray_ResolveWritebackIfCopy(fhatout_);
  Py_DECREF(fhatout_);

  free((char*)locdata);

  Py_INCREF(Py_None);
  return(Py_None);
  /* -- this is the end of the function if it executed normally -- */
}

/* 2D, 10x10 kernel interpolation for high accuracy
 * this version works with output points on a rectangular grid so that the same
 * weights in x and y can be used for many output points
 *
 * Inputs:
 *   infunc = input function on some grid, shape =(ngy,ngx)
 *   xpos = input x values, shape=(npi,nxo)
 *   ypos = input y values, shape=(npi,nyo)
 *
 * Outputs:
 * > fhatout = location to put the output values, shape=(npi,nyo*nxo)
 *
 * Notes:
 * there are npi*nyo*nxo interpolations to be done in total
 * but for each input pixel npi, there is an nyo x nxo grid of output points
 */
static PyObject *pyimcom_gridD5512C(PyObject *self, PyObject *args) {

  int i, j;
  long i_in, iy, ix, ipos;
  double **wx_ar, **wy_ar;
  double x, y;
  double xfh, yfh, *wx, *wy;
  long *xi, *yi; /* frac and integer parts of abscissae; note 'xfh' and 'yfh' will have 1/2 subtracted */
  long yip,xip;
  long ngy,ngx,npi,nxo,nyo;
  double xfh2, yfh2, e_, o_;
  double out, interp_vstrip;
  double *locdata, *L2;

  /* Input/output arrays */
  PyObject *infunc, *xpos, *ypos, *fhatout;
  PyArrayObject *infunc_, *xpos_, *ypos_, *fhatout_;

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
  ngy = infunc_->dimensions[0];
  ngx = infunc_->dimensions[1];
  npi = xpos_->dimensions[0];
  nxo = xpos_->dimensions[1];
  nyo = ypos_->dimensions[1];

  /* get local data */
  locdata = (double*)malloc((size_t)(ngy*ngx*sizeof(double)));
  ipos = 0;
  for(yip=0;yip<ngy;yip++)
    for(xip=0;xip<ngx;xip++)
      locdata[ipos++] = *(double*)PyArray_GETPTR2(infunc_,yip,xip);

  /* allocate arrays */
  wx_ar = (double**)malloc((size_t)(nxo*sizeof(double*)));
  for(ix=0;ix<nxo;ix++) wx_ar[ix] = (double*)malloc((size_t)(10*sizeof(double)));
  wy_ar = (double**)malloc((size_t)(nyo*sizeof(double*)));
  for(iy=0;iy<nyo;iy++) wy_ar[iy] = (double*)malloc((size_t)(10*sizeof(double)));
  xi = (long*)malloc((size_t)(nxo*sizeof(long)));
  yi = (long*)malloc((size_t)(nyo*sizeof(long)));

  /* loop over points to interpolate */
  for(i_in=0;i_in<npi;i_in++) {

    /* get the interpolation weights -- first in x, then in y.
     * do all the output points simultaneously to save time
     */
    for(ix=0;ix<nxo;ix++) {
      wx = wx_ar[ix];
      x = *(double*)PyArray_GETPTR2(xpos_,i_in,ix);
      xi[ix] = (long)floor(x);
      xfh = x-xi[ix]-.5;
      if (xi[ix]<4 || xi[ix]>=ngx-5) { /* point off the grid, don't interpolate */
        xi[ix]=4;
        for(j=0;j<10;j++) wx[j] = 0.;
        continue;
      }
      /* interpolation weights */
      xfh2 = xfh*xfh;
      e_ =  ((( 1.651881673372979740E-05*xfh2-3.145538007199505447E-04)*xfh2+1.793518183780194427E-03)*xfh2-2.904014557029917318E-03)*xfh2+6.187591260980151433E-04;
      o_ = ((((-3.486978652054735998E-06*xfh2+6.753750285320532433E-05)*xfh2-3.871378836550175566E-04)*xfh2+6.279918076641771273E-04)*xfh2-1.338434614116611838E-04)*xfh;
      wx[0] = e_ + o_;
      wx[9] = e_ - o_;
      e_ =  (((-1.146756217210629335E-04*xfh2+2.883845374976550142E-03)*xfh2-1.857047531896089884E-02)*xfh2+3.147734488597204311E-02)*xfh2-6.753293626461192439E-03;
      o_ = (((( 3.121412120355294799E-05*xfh2-8.040343683015897672E-04)*xfh2+5.209574765466357636E-03)*xfh2-8.847326408846412429E-03)*xfh2+1.898674086370833597E-03)*xfh;
      wx[1] = e_ + o_;
      wx[8] = e_ - o_;
      e_ =  ((( 3.256838096371517067E-04*xfh2-9.702063770653997568E-03)*xfh2+8.678848026470635524E-02)*xfh2-1.659182651092198924E-01)*xfh2+3.620560878249733799E-02;
      o_ = ((((-1.243658986204533102E-04*xfh2+3.804930695189636097E-03)*xfh2-3.434861846914529643E-02)*xfh2+6.581033749134083954E-02)*xfh2-1.436476114189205733E-02)*xfh;
      wx[2] = e_ + o_;
      wx[7] = e_ - o_;
      e_ =  (((-4.541830837949564726E-04*xfh2+1.494862093737218955E-02)*xfh2-1.668775957435094937E-01)*xfh2+5.879306056792649171E-01)*xfh2-1.367845996704077915E-01;
      o_ = (((( 2.894406669584551734E-04*xfh2-9.794291009695265532E-03)*xfh2+1.104231510875857830E-01)*xfh2-3.906954914039130755E-01)*xfh2+9.092432925988773451E-02)*xfh;
      wx[3] = e_ + o_;
      wx[6] = e_ - o_;
      e_ =  ((( 2.266560930061513573E-04*xfh2-7.815848920941316502E-03)*xfh2+9.686607348538181506E-02)*xfh2-4.505856722239036105E-01)*xfh2+6.067135256905490381E-01;
      o_ = ((((-4.336085507644610966E-04*xfh2+1.537862263741893339E-02)*xfh2-1.925091434770601628E-01)*xfh2+8.993141455798455697E-01)*xfh2-1.213035309579723942E+00)*xfh;
      wx[4] = e_ + o_;
      wx[5] = e_ - o_;
    }
    /* ... and now in y */
    for(iy=0;iy<nyo;iy++) {
      wy = wy_ar[iy];
      y = *(double*)PyArray_GETPTR2(ypos_,i_in,iy);
      yi[iy] = (long)floor(y);
      yfh = y-yi[iy]-.5;
      if (yi[iy]<4 || yi[iy]>=ngy-5) { /* point off the grid, don't interpolate */
        yi[iy]=4;
        for(j=0;j<10;j++) wy[j] = 0.;
        continue;
      }
      /* interpolation weights */
      yfh2 = yfh*yfh;
      e_ =  ((( 1.651881673372979740E-05*yfh2-3.145538007199505447E-04)*yfh2+1.793518183780194427E-03)*yfh2-2.904014557029917318E-03)*yfh2+6.187591260980151433E-04;
      o_ = ((((-3.486978652054735998E-06*yfh2+6.753750285320532433E-05)*yfh2-3.871378836550175566E-04)*yfh2+6.279918076641771273E-04)*yfh2-1.338434614116611838E-04)*yfh;
      wy[0] = e_ + o_;
      wy[9] = e_ - o_;
      e_ =  (((-1.146756217210629335E-04*yfh2+2.883845374976550142E-03)*yfh2-1.857047531896089884E-02)*yfh2+3.147734488597204311E-02)*yfh2-6.753293626461192439E-03;
      o_ = (((( 3.121412120355294799E-05*yfh2-8.040343683015897672E-04)*yfh2+5.209574765466357636E-03)*yfh2-8.847326408846412429E-03)*yfh2+1.898674086370833597E-03)*yfh;
      wy[1] = e_ + o_;
      wy[8] = e_ - o_;
      e_ =  ((( 3.256838096371517067E-04*yfh2-9.702063770653997568E-03)*yfh2+8.678848026470635524E-02)*yfh2-1.659182651092198924E-01)*yfh2+3.620560878249733799E-02;
      o_ = ((((-1.243658986204533102E-04*yfh2+3.804930695189636097E-03)*yfh2-3.434861846914529643E-02)*yfh2+6.581033749134083954E-02)*yfh2-1.436476114189205733E-02)*yfh;
      wy[2] = e_ + o_;
      wy[7] = e_ - o_;
      e_ =  (((-4.541830837949564726E-04*yfh2+1.494862093737218955E-02)*yfh2-1.668775957435094937E-01)*yfh2+5.879306056792649171E-01)*yfh2-1.367845996704077915E-01;
      o_ = (((( 2.894406669584551734E-04*yfh2-9.794291009695265532E-03)*yfh2+1.104231510875857830E-01)*yfh2-3.906954914039130755E-01)*yfh2+9.092432925988773451E-02)*yfh;
      wy[3] = e_ + o_;
      wy[6] = e_ - o_;
      e_ =  ((( 2.266560930061513573E-04*yfh2-7.815848920941316502E-03)*yfh2+9.686607348538181506E-02)*yfh2-4.505856722239036105E-01)*yfh2+6.067135256905490381E-01;
      o_ = ((((-4.336085507644610966E-04*yfh2+1.537862263741893339E-02)*yfh2-1.925091434770601628E-01)*yfh2+8.993141455798455697E-01)*yfh2-1.213035309579723942E+00)*yfh;
      wy[4] = e_ + o_;
      wy[5] = e_ - o_;
    }

    /* ... and now we can do the interpolation */
    ipos=0;
    for(iy=0;iy<nyo;iy++) { /* output pixel row */
      wy = wy_ar[iy];
      for(ix=0;ix<nxo;ix++) { /* output pixel column */
        wx = wx_ar[ix];
        out = 0.;
        L2 = locdata + (yi[iy]-4)*ngx + xi[ix]-4;
        for(i=0;i<10;i++) {
          interp_vstrip = 0.;
          for(j=0;j<10;j++) interp_vstrip += wx[j]*L2[j];
          out += interp_vstrip*wy[i];
          L2 += ngx;
        }
        *(double*)PyArray_GETPTR2(fhatout_, i_in, ipos++) = out;
      }
    }
  } /* end i_in loop */

  /* deallocate arrays */
  for(ix=0;ix<nxo;ix++) free((char*)wx_ar[ix]);
  free((char*)wx_ar);
  for(iy=0;iy<nyo;iy++) free((char*)wy_ar[iy]);
  free((char*)wy_ar);
  free((char*)xi);
  free((char*)yi);

  Py_DECREF(infunc_);
  Py_DECREF(xpos_);
  Py_DECREF(ypos_);
  PyArray_ResolveWritebackIfCopy(fhatout_);
  Py_DECREF(fhatout_);

  free((char*)locdata);

  Py_INCREF(Py_None);
  return(Py_None);
  /* -- this is the end of the function if it executed normally -- */
}

/*
 * ====== INTERPOLATION FUNCTIONS FOR DESTRIPING =======
 */

 /* Forward interpolation function
 *
 * Inputs:
 * image = pointer to the input image data (image to-be-interpolated; image "B")
 * g_eff = pointer to the input image pixel response matrix (image "B" g_eff)
 * rows, cols = dimensions of the images
 * coords = pointer to the array of coordinates (x,y) to be interpolated onto (image "A" coords)
 * num_coords = number of provided coordinate pairs

 *
 * Outputs:
 * > interpolated_image = pointer to output array for interpolated values. I_B interpolated onto I_A grid
 */
static PyObject *bilinear_interpolation(PyObject *self, PyObject *args) {
    int rows, cols, num_coords;
    long yip, xip, ipos;
    PyObject *image, *g_eff, *coords; /*inputs*/
    PyObject *interpolated_image; /*outputs*/
    PyArrayObject *image_, *g_eff_, *coords_; /*inputs*/
    PyArrayObject *interpolated_image_; /*outputs*/

      /* read arguments */
    if (!PyArg_ParseTuple(args, "O!O!iiO!iO!", &PyArray_Type, &image, &PyArray_Type, &g_eff, &rows, &cols,
    &PyArray_Type, &coords, &num_coords, &PyArray_Type, &interpolated_image)) {

    return(NULL);
    }
    /* repackage Python arrays as C objects */
    image_ = (PyArrayObject*)PyArray_FROM_OTF(image, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    g_eff_ = (PyArrayObject*)PyArray_FROM_OTF(g_eff, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    coords_ = (PyArrayObject*)PyArray_FROM_OTF(coords, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    interpolated_image_ = (PyArrayObject*)PyArray_FROM_OTF(interpolated_image, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (rows <= 0 || cols <= 0) {
        char error_msg[200];
        snprintf(error_msg, sizeof(error_msg),
                 "Invalid image dimensions: rows=%d, cols=%d", rows, cols);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return NULL;
    }

    // Get pointers to the data
    // double *image_data = *(double*)PyArray_GETPTR2(image_, imy, imx);
    // double *g_eff_data = *(double*)PyArray_GETPTR2(g_eff_, gy, gx);
    // double *coords_data = (double*)PyArray_DATA(coords_);
    //double *interp_data = *(double*)PyArray_GETPTR2(interpolated_image_, inty, intx);

    /* make local, flattened versions of arrays*/
    double *image_data = (double*)malloc((size_t)(cols*rows*sizeof(double)));
    double *g_eff_data = (double*)malloc((size_t)(cols*rows*sizeof(double)));
    double *coords_data = (double*)malloc((size_t)(cols*rows*sizeof(double)));
    double *interp_data = (double*)malloc((size_t)(cols*rows*sizeof(double)));
    ipos=0;
    for(yip=0;yip<cols;yip++) {
        for(xip=0;xip<rows;xip++) {
            image_data[ipos] = *(double*)PyArray_GETPTR2(image_, yip, xip);
            g_eff_data[ipos] = *(double*)PyArray_GETPTR2(g_eff_, yip, xip);
            coords_data[ipos] = *(double*)PyArray_GETPTR2(coords_,yip, xip);
            interp_data[ipos] = *(double*)PyArray_GETPTR2(interpolated_image_, yip, xip);
            ipos++;
          }
    }

    double x, y;
    int x1, y1, x2, y2;
    double dx, dy;

    for (int k = 0; k < num_coords; ++k) { //iterate through coordinate pairs
         y = coords_data[2*k];
         x = coords_data[2*k+1];

        // Calculate the indices of the four surrounding pixels
         x1 = (int)floor(x);
         y1 = (int)floor(y);
         x2 = x1 + 1;
         y2 = y1 + 1;

        if (x1 < 0 || x2 >= cols || y1 < 0 || y2 >= rows || x1 >= cols || y1 >= rows) {
            continue; // Skip out-of-bounds; Might want to change this later?
        }

        // Compute fractional distances from x1 and y1
         dx = x - x1;
         dy = y - y1;
         int idx11 = y1 * cols + x1;

         if(k<=5){
            printf("dimensions: x=%f, y=%f, x1=%d, y1=%d, x2=%d, y2=%d, dx=%f, dy=%f, k=%d, idx11=%d",
                 x, y, x1, y1, x2, y2, dx, dy, k, idx11);
            fflush(stdout);
         }

        // Compute contributions; image_A_interp[pixel] = (weight)*(image_B[contributing_pixel])*(g_eff_B[contributing_pixel])
        interp_data[k] =
            (1. - dx) * (1. - dy) * image_data[y1 * cols + x1] * g_eff_data[y1 * cols + x1]
            + (1. - dx) * dy * image_data[y2 * cols + x1] * g_eff_data[y2 * cols + x1]
            + (1. - dy) * dx * image_data[y1 * cols + x2] * g_eff_data[y1 * cols + x2]
            + dx * dy * image_data[y2 * cols + x2] * g_eff_data[y2 * cols + x2];


    }  //end iteration over coordinate pairs

    // Copy results back to numpy array
    for (int k = 0; k < num_coords; ++k) { //iterate through coordinate pairs
         int yk = k / cols;
         int xk = k % cols;
        *(double*)PyArray_GETPTR2(interpolated_image_, yk, xk) = interp_data[k];
    }

     /* reference count and resolve */
    Py_DECREF(image_);
    Py_DECREF(g_eff_);
    Py_DECREF(coords_);
    PyArray_ResolveWritebackIfCopy(interpolated_image_);
    Py_DECREF(interpolated_image_);

    free((char*)image_data);
    free((char*)g_eff_data);
    free((char*)coords_data);
    free((char*)interp_data);

    Py_INCREF(Py_None);

    return(Py_None);

} /* end static pyobject bilinear interpolation */


 /*  Transpose interpolation function
 *
 * Inputs:
 * image = pointer to the gradient image data (gradient image to-be-transpose-interpolated; image "gradient_interpolated")
 * rows, cols = dimensions of the image
 * coords = pointer to the array of coordinates (x,y) to be interpolated onto (image "B" coords)
 * num_coords = number of provided coordinate pairs

 *
 * Outputs:
 * > original_image = pointer for the output of transpose interpolation (gradient image interpolated onto image "B" grid)
*/

static PyObject *bilinear_transpose (PyObject *self, PyObject *args){
    int rows, cols, num_coords;
    PyObject *image, *coords; /*inputs*/
    PyObject *original_image; /*outputs*/
    PyArrayObject *image_, *coords_; /*inputs*/
    PyArrayObject *original_image_; /*outputs*/
    PyArrayObject *weight_image_;

      /* read arguments */
    if (!PyArg_ParseTuple(args, "O!iiO!iO!", &PyArray_Type, &image, &rows, &cols,
    &PyArray_Type, &coords, &num_coords, &PyArray_Type, &original_image)) {

    return(NULL);
    }
    /* repackage Python arrays as C objects */
    image_ = (PyArrayObject*)PyArray_FROM_OTF(image, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    coords_ = (PyArrayObject*)PyArray_FROM_OTF(coords, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    original_image_ = (PyArrayObject*)PyArray_FROM_OTF(original_image, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    weight_image_ = (PyArrayObject*)PyArray_ZEROS(2, PyArray_DIMS(original_image_), NPY_DOUBLE, 0);


    if (rows <= 0 || cols <= 0) {
        char error_msg[200];
        snprintf(error_msg, sizeof(error_msg),
                 "Invalid image dimensions: rows=%d, cols=%d", rows, cols);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return NULL;
    }

    // Get pointers to the data
    double *image_data = (double*)PyArray_DATA(image_);
    double *coords_data = (double*)PyArray_DATA(coords_);
    double *original_data = (double*)PyArray_DATA(original_image_);
    double *weight_data = (double*)PyArray_DATA(weight_image_);


    double x, y;
    int x1, y1, x2, y2;
    double dx, dy;

    for (int k = 0; k < num_coords; ++k) {
        x = coords_data[2 * k];
        y = coords_data[2 * k + 1];

        x1 = (int)floor(x);
        y1 = (int)floor(y);
        x2 = x1 + 1;
        y2 = y1 + 1;

        if (x1 < 0 || x2 >= cols || y1 < 0 || y2 >= rows) {
            continue; // Skip out-of-bounds
        }

        // Compute fractional distances from x1 and y1
        dx = x - x1;
        dy = y - y1;

        // Weights
        double w11 = (1 - dx) * (1 - dy);
        double w12 = (1 - dx) * dy;
        double w21 = dx * (1 - dy);
        double w22 = dx * dy;

        // Accumulate contributions based on weights
        original_data[y1 * cols + x1] += w11 * image_data[k];
        original_data[y1 * cols + x2] += w12 * image_data[k];
        original_data[y2 * cols + x1] += w21 * image_data[k];
        original_data[y2 * cols + x2] += w22 * image_data[k];

        // Weight map
        weight_data[y1 * cols + x1] += w11;
        weight_data[y1 * cols + x2] += w12;
        weight_data[y2 * cols + x1] += w21;
        weight_data[y2 * cols + x2] += w22;

    }

    for (int i = 0; i < rows * cols; ++i) {
        if (weight_data[i] > 0) {
            original_data[i] /= weight_data[i];
        }
    }
    /* reference count and resolve */
    Py_DECREF(image_);
    Py_DECREF(coords_);
    Py_DECREF(weight_image_);
    PyArray_ResolveWritebackIfCopy(original_image_);
    Py_DECREF(original_image_);

    Py_INCREF(Py_None);

    return(Py_None);
}
/* void bilinear_transpose(float* image, int rows, int cols, float* coords, int num_coords, float* original_image) {

    for (int k = 0; k < num_coords; ++k) {
        float x = coords[2 * k];
        float y = coords[2 * k + 1];

        int x1 = (int)floor(x);
        int y1 = (int)floor(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        if (x1 < 0 || x2 >= cols || y1 < 0 || y2 >= rows) {
            continue; // Skip out-of-bounds
        }

        // Compute fractional distances from x1 and y1
        float dx = x - x1;
        float dy = y - y1;

        // Accumulate contributions based on weights
        original_image[y1 * cols + x1] += (1 - dx) * (1 - dy) * image[k] ;
        original_image[y1 * cols + x2] += (1 - dx) * dy * image[k] ;
        original_image[y2 * cols + x1] += dx * (1 - dy) * image[k] ;
        original_image[y2 * cols + x2] += dx * dy * image[k] ;
    }
}
end original transpose interpolation*/


/*
 * ===== ROUTINES FOR KAPPA INTERPOLATION =====
 */

/* routine to solve Ax=b, for:
 * A = positive definite matrix (NxN)
 * x = vector, output (N)
 * b = vector (N)
 * only the lower triangle of A is ever used (the rest need not even be allocated). The matrix A is destroyed.
 */
void pyimcom_lsolve_sps(int N, double **A, double *x, double *b) {
  int i, j, k;
  double sum;
  double *p1, *p2;

  /* Replace A with its Cholesky decomposition */
  for(i=0;i<N;i++) {
    p1 = A[i];
    for(j=0;j<i;j++) {
      p2 = A[j];
      sum = 0.;
      for(k=0;k<j;k++) sum+= p1[k]*p2[k];
      p1[j] = (p1[j]-sum)/p2[j];
    }
    sum = 0.;
    for(k=0;k<i;k++) sum+= p1[k]*p1[k];
    p1[i] = sqrt(p1[i]-sum);
  }
  /* ... now the lower part of A is the Cholesky decomposition L: A = LL^T */

  /* now get p1 = LT-1 b */
  p1 = (double*)malloc((size_t)(N*sizeof(double)));
  for(i=0;i<N;i++) {
    sum = 0.;
    p2 = A[i];
    for(j=0;j<i;j++) sum += p2[j]*p1[j];
    p1[i] = (b[i]-sum)/p2[i];
  }
  /* ... and x = L^-1 p1 */
  for(i=N-1;i>=0;i--) {
    sum = 0.;
    for(j=i+1;j<N;j++) sum += A[j][i]*x[j];
    x[i] = (p1[i]-sum)/A[i][i];
  }

  free((char*)p1);
}

/* Makes coaddition matrix T from a reduced space.
 *
 * Inputs:
 *   nv = number of 'node' eigenvalues (must be >=2)
 *   m = number of output pixels
 *   Nflat = input noise array, shape (m,nv,nv).flatten
 *   Dflat = input 1st order signal D/C, shape (m,nv).flatten
 *   Eflat = input 2nd order signal E/C, shape (m,nv,nv).flatten
 *   kappa = list of eigenvalues, must be sorted ascending, shape (nv)
 *   ucmin = min U/C
 *   smax = max Sigma (noise)
 *
 * Outputs:
 * > out_kappa = output "kappa" parameter, shape (m)
 * > out_Sigma = output "Sigma", shape (m)
 * > out_UC = output "U/C", shape (m)
 * > out_w = output weights for each eigenvalue and each output pixel, shape (m,nv).flatten
 */
void pyimcom_build_reduced_T(int nv, long m, double *Nflat, double *Dflat, double *Eflat,
  double *kappa, double ucmin, double smax, double *out_kappa, double *out_Sigma, double *out_UC, double *out_w) {

  int iv, jv, nv2, ik;
  long a;
  double S, UC, sum;
  double *Nflat_a, *Dflat_a, *Eflat_a;
  double kappamid, factor;
  double *w, **M2d;

  /* allocate memory */
  M2d = (double**)malloc((size_t)(nv*sizeof(double*)));
  M2d[0] = (double*)malloc((size_t)(nv*(nv+1)/2*sizeof(double)));
  for(iv=1;iv<nv;iv++) M2d[iv] = M2d[iv-1]+iv;

  /* set up pointers to inputs and outputs */
  nv2 = nv*nv;
  Nflat_a = Nflat;
  Dflat_a = Dflat;
  Eflat_a = Eflat;

  /* loop over output pixels */
  for(a=0;a<m;a++) {

    w = out_w + a*nv; /* pointer to weights for this pixel */

    /* first figure out the range of kappa */
    iv=nv-1;
    do {
      iv--;
      S = Nflat_a[iv*(nv+1)]; /* diagonal noises */
      UC = 1. - 2*Dflat_a[iv] + Eflat_a[iv*(nv+1)]; /* diagonal U/C */
    } while (iv>0 && UC>ucmin && S<smax);
    /* kappa should be in the range kappa[iv] .. kappa[iv+1] */
    kappamid = sqrt(kappa[iv]*kappa[iv+1]);
    factor = pow(kappa[iv+1]/kappa[iv], .25);

    /* iterative loop to find 'best' kappa */
    for(ik=0;ik<12;ik++) {

      /* build matrix for this kappa */
      for(iv=0;iv<nv;iv++)
        for(jv=0;jv<=iv;jv++)
          M2d[iv][jv] = Eflat_a[iv+nv*jv] + kappamid*Nflat_a[iv+nv*jv];
      /* ... and get weights */
      pyimcom_lsolve_sps(nv, M2d, w, Dflat_a);

      /* now get the UC and the S */
      S = 0.;
      for(iv=0;iv<nv;iv++) {
        sum = 0.;
        for(jv=0;jv<nv;jv++) sum += Nflat_a[iv+nv*jv]*w[jv];
        S += w[iv]*sum;
      }
      UC = 1.-kappamid*S;
      for(iv=0;iv<nv;iv++) UC -= Dflat_a[iv]*w[iv];

      /* updates to kappa */
      kappamid *= UC>ucmin && S<smax? 1./factor: factor;
      factor = sqrt(factor);
    }

    /* output other information */
    out_kappa[a] = kappamid;
    out_Sigma[a] = S;
    out_UC[a] = UC;

    /* increment pointers to portions of inputs and outputs */
    Nflat_a += nv2;
    Dflat_a += nv;
    Eflat_a += nv2;
  }

  free((char*)M2d[0]);
  free((char*)M2d);
}
/* ... and here is the Python wrapper.
 * No need to give nv and m, so:
 *
 * Inputs:
 *   Nflat = input noise array, shape (m,nv,nv).flatten
 *   Dflat = input 1st order signal D/C, shape (m,nv).flatten
 *   Eflat = input 2nd order signal E/C, shape (m,nv,nv).flatten
 *   kappa = list of eigenvalues, must be sorted ascending, shape (nv)
 *   ucmin = min U/C
 *   smax = max Sigma (noise)
 *
 * Outputs:
 * > out_kappa = output "kappa" parameter, shape (m)
 * > out_Sigma = output "Sigma", shape (m)
 * > out_UC = output "U/C", shape (m)
 * > out_w = output weights for each eigenvalue and each output pixel, shape (m,nv).flatten
 */
static PyObject *pyimcom_build_reduced_T_wrap(PyObject *self, PyObject *args) {

  int nv;
  long m, mnv, mnv2, im;
  PyObject *Nflat, *Dflat, *Eflat, *kappa; /* inputs */
  double ucmin, smax;
  PyObject *out_kappa, *out_Sigma, *out_UC, *out_w; /* outputs */
  PyArrayObject *Nflat_, *Dflat_, *Eflat_, *kappa_, *out_kappa_, *out_Sigma_, *out_UC_, *out_w_;
  double *vec;
  double *myNflat, *myDflat, *myEflat, *mykappa, *myout_kappa, *myout_Sigma, *myout_UC, *myout_w;

  /* read arguments */
  if (!PyArg_ParseTuple(args, "O!O!O!O!ddO!O!O!O!",
    &PyArray_Type, &Nflat, &PyArray_Type, &Dflat, &PyArray_Type, &Eflat, &PyArray_Type, &kappa, &ucmin, &smax,
    &PyArray_Type, &out_kappa, &PyArray_Type, &out_Sigma, &PyArray_Type, &out_UC, &PyArray_Type, &out_w)) {

    return(NULL);
  }

  /* repackage Python arrays as C objects */
  Nflat_ = (PyArrayObject*)PyArray_FROM_OTF(Nflat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  Dflat_ = (PyArrayObject*)PyArray_FROM_OTF(Dflat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  Eflat_ = (PyArrayObject*)PyArray_FROM_OTF(Eflat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  kappa_ = (PyArrayObject*)PyArray_FROM_OTF(kappa, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  out_kappa_ = (PyArrayObject*)PyArray_FROM_OTF(out_kappa, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  out_Sigma_ = (PyArrayObject*)PyArray_FROM_OTF(out_Sigma, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  out_UC_ = (PyArrayObject*)PyArray_FROM_OTF(out_UC, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  out_w_ = (PyArrayObject*)PyArray_FROM_OTF(out_w, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  /* dimensions */
  nv = kappa_->dimensions[0];
  m = out_kappa_->dimensions[0];

  /* package inputs into flat C arrays to minimize pointer arithmetic in inner for loops */
  vec = (double*)malloc((size_t)((m*nv*nv*2+m*nv*2+nv+m*3)*sizeof(double)));
  mnv = m*nv; mnv2 = mnv*nv;
  myNflat = vec;
  myDflat = myNflat + mnv2;
  myEflat = myDflat + mnv;
  mykappa = myEflat + mnv2;
  myout_kappa = mykappa + nv;
  myout_Sigma = myout_kappa + m;
  myout_UC = myout_Sigma + m;
  myout_w = myout_UC + m;
  for(im=0;im<mnv2;im++) myNflat[im] = *(double*)PyArray_GETPTR1(Nflat_,im);
  for(im=0;im<mnv;im++) myDflat[im] = *(double*)PyArray_GETPTR1(Dflat_,im);
  for(im=0;im<mnv2;im++) myEflat[im] = *(double*)PyArray_GETPTR1(Eflat_,im);
  for(im=0;im<nv;im++) mykappa[im] = *(double*)PyArray_GETPTR1(kappa_,im);

  /* OK, now the C function call! */
  pyimcom_build_reduced_T(nv,m,myNflat,myDflat,myEflat,mykappa,ucmin,smax,myout_kappa,myout_Sigma,myout_UC,myout_w);

  /* package outputs */
  for(im=0;im<m;im++) {
    *(double*)PyArray_GETPTR1(out_kappa_,im) = myout_kappa[im];
    *(double*)PyArray_GETPTR1(out_Sigma_,im) = myout_Sigma[im];
    *(double*)PyArray_GETPTR1(out_UC_,im) = myout_UC[im];
  }
  for(im=0;im<mnv;im++) *(double*)PyArray_GETPTR1(out_w_,im) = myout_w[im];
  PyArray_ResolveWritebackIfCopy(out_kappa_);
  Py_DECREF(out_kappa_);
  PyArray_ResolveWritebackIfCopy(out_Sigma_);
  Py_DECREF(out_Sigma_);
  PyArray_ResolveWritebackIfCopy(out_UC_);
  Py_DECREF(out_UC_);
  PyArray_ResolveWritebackIfCopy(out_w_);
  Py_DECREF(out_w_);

  /* cleanup and return */
  free((char*)vec);
  Py_INCREF(Py_None);
  return(Py_None);
}

void pyimcom_test_sps(void) {
  FILE *fp;
  double **A, *x, *b;
  int i,j,N,nv,iv;
  long pa,m,ia;
  double *Nf, *Df, *Ef, *ka, *ka_out, *S, *UC, *w;
  double sum;

  /* test of matrix inversion */

  N = 4;
  A = (double**)malloc((size_t)(N*sizeof(double*)));
  for(i=0;i<N;i++) A[i] = (double*)malloc((size_t)(N*sizeof(double)));
  x = (double*)malloc((size_t)(2*N*sizeof(double)));
  b = x+N;

  for(i=0;i<N;i++) for(j=0;j<N;j++) A[i][j] = sqrt(i+j);
  for(i=0;i<N;i++) A[i][i] += N*N;
  for(i=0;i<N;i++) b[i] = (8-i)*(8-i);

  pyimcom_lsolve_sps(N,A,x,b);
  for(i=0;i<N;i++) {
    for(j=0;j<N;j++)
      printf(" %12.5lE", A[i][j]);
    printf("  |  %12.5lE\n", x[i]);
  }

  for(i=0;i<N;i++) free((char*)A[i]);
  free((char*)A);
  free((char*)x);

  /* test of node interpolation */
  fp = fopen("matrixdata.txt", "r");
  fscanf(fp, "%ld %d", &m, &nv);
  Nf = (double*)malloc((size_t)(m*nv*nv*sizeof(double)));
  Df = (double*)malloc((size_t)(m*nv*sizeof(double)));
  Ef = (double*)malloc((size_t)(m*nv*nv*sizeof(double)));
  ka = (double*)malloc((size_t)(nv*sizeof(double)));
  w = (double*)malloc((size_t)(m*nv*sizeof(double)));
  ka_out = (double*)malloc((size_t)(3*m*sizeof(double)));
  S = ka_out+m;
  UC = ka_out+2*m;
  /* read data */
  for(ia=0;ia<m*nv*nv;ia++) fscanf(fp, "%lg", Nf+ia);
  for(ia=0;ia<m*nv;ia++) fscanf(fp, "%lg", Df+ia);
  for(ia=0;ia<m*nv*nv;ia++) fscanf(fp, "%lg", Ef+ia);
  for(ia=0;ia<nv;ia++) fscanf(fp, "%lg", ka+ia);
  fclose(fp);

  /* write some info */
  printf("%ld pixels, %d eigenvalue nodes\n", m, nv);
  printf("kappa ->");
  for(i=0;i<nv;i++) printf(" %16.9lE", ka[i]);
  printf("\n\n");

  pyimcom_build_reduced_T(nv, m, Nf, Df, Ef, ka, 1e-6, 0.6, ka_out, S, UC, w);

  for(pa=0;pa<25;pa++) {
    printf("%5ld %14.7lE %14.7lE %14.7lE    ", pa, ka_out[pa], S[pa], UC[pa]);
    sum = 0.;
    for(iv=0;iv<nv;iv++) {
      printf(" %10.6lf", w[pa*nv+iv]);
      sum += w[pa*nv+iv];
    }
    printf("      %10.6lf\n", sum);
  }

  free((char*)Nf);
  free((char*)Df);
  free((char*)Ef);
  free((char*)ka);
  free((char*)ka_out);
  free((char*)w);
}

/* Method Table */
static PyMethodDef PyImcom_CMethods[] = {
  {"lakernel1", (PyCFunction)pyimcom_lakernel1, METH_VARARGS, "PyIMCOM core linear algebra kernel"},
  {"iD5512C", (PyCFunction)pyimcom_iD5512C, METH_VARARGS, "interpolation routine"},
  {"iD5512C_sym", (PyCFunction)pyimcom_iD5512C_sym, METH_VARARGS, "interpolation routine"},
  {"gridD5512C", (PyCFunction)pyimcom_gridD5512C, METH_VARARGS, "interpolation routine regular grid"},
  {"build_reduced_T_wrap", (PyCFunction)pyimcom_build_reduced_T_wrap, METH_VARARGS, "fast approximate coadd matrix"},
  {"bilinear_interpolation", (PyCFunction)bilinear_interpolation, METH_VARARGS, "Interpolate image B onto image A"},
  {"bilinear_transpose", (PyCFunction)bilinear_transpose, METH_VARARGS, "Transpose interpolation"},
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

