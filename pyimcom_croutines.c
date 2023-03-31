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

  int i,j,k;
  long nlayer, nout, ngy, ngx, ipos, ilayer;
  double x, y, xfh, yfh; long xi, yi; /* frac and integer parts of abscissae; note 'xfh' and 'yfh' will have 1/2 subtracted */
  double wx[21], wy[21], *wwx, *wwy;
  double cx,sx,cy,sy,u;
  double interp_vstrip, out;
  long ds,ds2;
  char *temp;

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
  /* .. and strides .. */
  ds = infunc_->strides[2];
  ds2 = infunc_->strides[1] - 10*ds;

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
    wwx=wx+10; wwy=wy+10;
    cx = cos(u=7.795042160878816462e-02*xfh);
    sx = sin(u);
    cy = cos(u=7.795042160878816462e-02*yfh);
    sy = sin(u);
    *wwx++ = 1912.40267850101*cx;
    *wwy++ = 1912.40267850101*cy;
    *wwx++ = -12176.9938608718*cx;
    *wwy++ = -12176.9938608718*cy;
    *wwx++ = 32463.3012682714*cx;
    *wwy++ = 32463.3012682714*cy;
    *wwx++ = -43429.0459588089*cx;
    *wwy++ = -43429.0459588089*cy;
    *wwx++ = 21230.9659767686*cx;
    *wwy++ = 21230.9659767686*cy;
    *wwx++ = -49042.3061911076*sx;
    *wwy++ = -49042.3061911076*sy;
    *wwx++ = 410355.593860622*sx;
    *wwy++ = 410355.593860622*sy;
    *wwx++ = -1555126.53918249*sx;
    *wwy++ = -1555126.53918249*sy;
    *wwx++ = 3501335.76171435*sx;
    *wwy++ = 3501335.76171435*sy;
    *wwx++ = -5159499.1491332*sx;
    *wwy++ = -5159499.1491332*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(u=2.269252977160159945e-01*xfh);
    sx = sin(u);
    cy = cos(u=2.269252977160159945e-01*yfh);
    sy = sin(u);
    *wwx++ += -4927.00410046915*cx;
    *wwy++ += -4927.00410046915*cy;
    *wwx++ += 31594.8125350013*cx;
    *wwy++ += 31594.8125350013*cy;
    *wwx++ += -84620.6445366486*cx;
    *wwy++ += -84620.6445366486*cy;
    *wwx++ += 113528.194562436*cx;
    *wwy++ += 113528.194562436*cy;
    *wwx++ += -55575.5507391098*cx;
    *wwy++ += -55575.5507391098*cy;
    *wwx++ += 43237.5141237444*sx;
    *wwy++ += 43237.5141237444*sy;
    *wwx++ += -363739.086758674*sx;
    *wwy++ += -363739.086758674*sy;
    *wwx++ += 1383601.73837127*sx;
    *wwy++ += 1383601.73837127*sy;
    *wwx++ += -3122480.58932711*sx;
    *wwy++ += -3122480.58932711*sy;
    *wwx++ += 4606470.74368757*sx;
    *wwy++ += 4606470.74368757*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(u=3.557380180911379752e-01*xfh);
    sx = sin(u);
    cy = cos(u=3.557380180911379752e-01*yfh);
    sy = sin(u);
    *wwx++ += 5835.90561316373*cx;
    *wwy++ += 5835.90561316373*cy;
    *wwx++ += -37854.7594904635*cx;
    *wwy++ += -37854.7594904635*cy;
    *wwx++ += 102189.161922347*cx;
    *wwy++ += 102189.161922347*cy;
    *wwx++ += -137779.981708223*cx;
    *wwy++ += -137779.981708223*cy;
    *wwx++ += 67609.7669203679*cx;
    *wwy++ += 67609.7669203679*cy;
    *wwx++ += -32463.3907553235*sx;
    *wwy++ += -32463.3907553235*sy;
    *wwx++ += 275501.443053076*sx;
    *wwy++ += 275501.443053076*sy;
    *wwx++ += -1054523.73112156*sx;
    *wwy++ += -1054523.73112156*sy;
    *wwx++ += 2389400.02886295*sx;
    *wwy++ += 2389400.02886295*sy;
    *wwx++ += -3531921.53431643*sx;
    *wwy++ += -3531921.53431643*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(u=4.529461196132943956e-01*xfh);
    sx = sin(u);
    cy = cos(u=4.529461196132943956e-01*yfh);
    sy = sin(u);
    *wwx++ += -4322.72244949997*cx;
    *wwy++ += -4322.72244949997*cy;
    *wwx++ += 28369.9538060603*cx;
    *wwy++ += 28369.9538060603*cy;
    *wwx++ += -77242.1292497577*cx;
    *wwy++ += -77242.1292497577*cy;
    *wwx++ += 104725.479751602*cx;
    *wwy++ += 104725.479751602*cy;
    *wwx++ += -51530.6247479867*cx;
    *wwy++ += -51530.6247479867*cy;
    *wwx++ += 18759.6895246111*sx;
    *wwy++ += 18759.6895246111*sy;
    *wwx++ += -160638.901862404*sx;
    *wwy++ += -160638.901862404*sy;
    *wwx++ += 618972.720736964*sx;
    *wwy++ += 618972.720736964*sy;
    *wwx++ += -1408673.16724246*sx;
    *wwy++ += -1408673.16724246*sy;
    *wwx++ += 2086791.19148948*sx;
    *wwy++ += 2086791.19148948*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(u=5.099362658787808256e-01*xfh);
    sx = sin(u);
    cy = cos(u=5.099362658787808256e-01*yfh);
    sy = sin(u);
    *wwx++ += 1501.41887706351*cx;
    *wwy++ += 1501.41887706351*cy;
    *wwx++ += -9933.01974301992*cx;
    *wwy++ += -9933.01974301992*cy;
    *wwx++ += 27210.3468013967*cx;
    *wwy++ += 27210.3468013967*cy;
    *wwx++ += -37044.7834316057*cx;
    *wwy++ += -37044.7834316057*cy;
    *wwx++ += 18266.0493034857*cx;
    *wwy++ += 18266.0493034857*cy;
    *wwx++ += -5760.49192550392*sx;
    *wwy++ += -5760.49192550392*sy;
    *wwx++ += 49630.9882615042*sx;
    *wwy++ += 49630.9882615042*sy;
    *wwx++ += -192138.896175493*sx;
    *wwy++ += -192138.896175493*sy;
    *wwx++ += 438666.473889707*sx;
    *wwy++ += 438666.473889707*sy;
    *wwx++ += -650877.476593785*sx;
    *wwy++ += -650877.476593785*sy;
    wx[ 0] = wx[10] + wx[15];
    wx[ 1] = wx[11] + wx[16];
    wx[ 2] = wx[12] + wx[17];
    wx[ 3] = wx[13] + wx[18];
    wx[ 4] = wx[14] + wx[19];
    wx[ 5] = wx[14] - wx[19];
    wx[ 6] = wx[13] - wx[18];
    wx[ 7] = wx[12] - wx[17];
    wx[ 8] = wx[11] - wx[16];
    wx[ 9] = wx[10] - wx[15];
    wy[ 0] = wy[10] + wy[15];
    wy[ 1] = wy[11] + wy[16];
    wy[ 2] = wy[12] + wy[17];
    wy[ 3] = wy[13] + wy[18];
    wy[ 4] = wy[14] + wy[19];
    wy[ 5] = wy[14] - wy[19];
    wy[ 6] = wy[13] - wy[18];
    wy[ 7] = wy[12] - wy[17];
    wy[ 8] = wy[11] - wy[16];
    wy[ 9] = wy[10] - wy[15];
    /* end loop unrolled code generated by perl */

    /* and the outputs */
    for(ilayer=0;ilayer<nlayer;ilayer++) {
      out = 0.;
      temp = (char*)PyArray_GETPTR3(infunc_,ilayer,yi-4,xi-4); /* set temp to point to corner of interpolation region */
      for(i=0;i<10;i++) {
        interp_vstrip = 0.;
        for(j=0;j<10;j++) {
          interp_vstrip += wx[j]*(*(double*)temp);
          temp += ds;
        }
        out += interp_vstrip*wy[i];
        temp += ds2; /* jump to next row of input image */
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
  double wx[21], wy[21], *wwx, *wwy;
  double cx,sx,cy,sy;
  double interp_vstrip, out;
  long ds,ds2;
  char *temp;
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
  /* .. and strides .. */
  ds = infunc_->strides[2];
  ds2 = infunc_->strides[1] - 10*ds;

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

    /* now compute the weights */
    wwx=wx+10; wwy=wy+10;
    cx = cos(7.795042160878816462e-02*xfh);
    cy = cos(7.795042160878816462e-02*yfh);
    *wwx++ = 1912.40267850101*cx;
    *wwy++ = 1912.40267850101*cy;
    *wwx++ = -12176.9938608718*cx;
    *wwy++ = -12176.9938608718*cy;
    *wwx++ = 32463.3012682714*cx;
    *wwy++ = 32463.3012682714*cy;
    *wwx++ = -43429.0459588089*cx;
    *wwy++ = -43429.0459588089*cy;
    *wwx++ = 21230.9659767686*cx;
    *wwy++ = 21230.9659767686*cy;
    sx = sin(7.795042160878816462e-02*xfh);
    sy = sin(7.795042160878816462e-02*yfh);
    *wwx++ = -49042.3061911076*sx;
    *wwy++ = -49042.3061911076*sy;
    *wwx++ = 410355.593860622*sx;
    *wwy++ = 410355.593860622*sy;
    *wwx++ = -1555126.53918249*sx;
    *wwy++ = -1555126.53918249*sy;
    *wwx++ = 3501335.76171435*sx;
    *wwy++ = 3501335.76171435*sy;
    *wwx++ = -5159499.1491332*sx;
    *wwy++ = -5159499.1491332*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(2.269252977160159945e-01*xfh);
    cy = cos(2.269252977160159945e-01*yfh);
    *wwx++ += -4927.00410046915*cx;
    *wwy++ += -4927.00410046915*cy;
    *wwx++ += 31594.8125350013*cx;
    *wwy++ += 31594.8125350013*cy;
    *wwx++ += -84620.6445366486*cx;
    *wwy++ += -84620.6445366486*cy;
    *wwx++ += 113528.194562436*cx;
    *wwy++ += 113528.194562436*cy;
    *wwx++ += -55575.5507391098*cx;
    *wwy++ += -55575.5507391098*cy;
    sx = sin(2.269252977160159945e-01*xfh);
    sy = sin(2.269252977160159945e-01*yfh);
    *wwx++ += 43237.5141237444*sx;
    *wwy++ += 43237.5141237444*sy;
    *wwx++ += -363739.086758674*sx;
    *wwy++ += -363739.086758674*sy;
    *wwx++ += 1383601.73837127*sx;
    *wwy++ += 1383601.73837127*sy;
    *wwx++ += -3122480.58932711*sx;
    *wwy++ += -3122480.58932711*sy;
    *wwx++ += 4606470.74368757*sx;
    *wwy++ += 4606470.74368757*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(3.557380180911379752e-01*xfh);
    cy = cos(3.557380180911379752e-01*yfh);
    *wwx++ += 5835.90561316373*cx;
    *wwy++ += 5835.90561316373*cy;
    *wwx++ += -37854.7594904635*cx;
    *wwy++ += -37854.7594904635*cy;
    *wwx++ += 102189.161922347*cx;
    *wwy++ += 102189.161922347*cy;
    *wwx++ += -137779.981708223*cx;
    *wwy++ += -137779.981708223*cy;
    *wwx++ += 67609.7669203679*cx;
    *wwy++ += 67609.7669203679*cy;
    sx = sin(3.557380180911379752e-01*xfh);
    sy = sin(3.557380180911379752e-01*yfh);
    *wwx++ += -32463.3907553235*sx;
    *wwy++ += -32463.3907553235*sy;
    *wwx++ += 275501.443053076*sx;
    *wwy++ += 275501.443053076*sy;
    *wwx++ += -1054523.73112156*sx;
    *wwy++ += -1054523.73112156*sy;
    *wwx++ += 2389400.02886295*sx;
    *wwy++ += 2389400.02886295*sy;
    *wwx++ += -3531921.53431643*sx;
    *wwy++ += -3531921.53431643*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(4.529461196132943956e-01*xfh);
    cy = cos(4.529461196132943956e-01*yfh);
    *wwx++ += -4322.72244949997*cx;
    *wwy++ += -4322.72244949997*cy;
    *wwx++ += 28369.9538060603*cx;
    *wwy++ += 28369.9538060603*cy;
    *wwx++ += -77242.1292497577*cx;
    *wwy++ += -77242.1292497577*cy;
    *wwx++ += 104725.479751602*cx;
    *wwy++ += 104725.479751602*cy;
    *wwx++ += -51530.6247479867*cx;
    *wwy++ += -51530.6247479867*cy;
    sx = sin(4.529461196132943956e-01*xfh);
    sy = sin(4.529461196132943956e-01*yfh);
    *wwx++ += 18759.6895246111*sx;
    *wwy++ += 18759.6895246111*sy;
    *wwx++ += -160638.901862404*sx;
    *wwy++ += -160638.901862404*sy;
    *wwx++ += 618972.720736964*sx;
    *wwy++ += 618972.720736964*sy;
    *wwx++ += -1408673.16724246*sx;
    *wwy++ += -1408673.16724246*sy;
    *wwx++ += 2086791.19148948*sx;
    *wwy++ += 2086791.19148948*sy;
    wwx=wx+10; wwy=wy+10;
    cx = cos(5.099362658787808256e-01*xfh);
    cy = cos(5.099362658787808256e-01*yfh);
    *wwx++ += 1501.41887706351*cx;
    *wwy++ += 1501.41887706351*cy;
    *wwx++ += -9933.01974301992*cx;
    *wwy++ += -9933.01974301992*cy;
    *wwx++ += 27210.3468013967*cx;
    *wwy++ += 27210.3468013967*cy;
    *wwx++ += -37044.7834316057*cx;
    *wwy++ += -37044.7834316057*cy;
    *wwx++ += 18266.0493034857*cx;
    *wwy++ += 18266.0493034857*cy;
    sx = sin(5.099362658787808256e-01*xfh);
    sy = sin(5.099362658787808256e-01*yfh);
    *wwx++ += -5760.49192550392*sx;
    *wwy++ += -5760.49192550392*sy;
    *wwx++ += 49630.9882615042*sx;
    *wwy++ += 49630.9882615042*sy;
    *wwx++ += -192138.896175493*sx;
    *wwy++ += -192138.896175493*sy;
    *wwx++ += 438666.473889707*sx;
    *wwy++ += 438666.473889707*sy;
    *wwx++ += -650877.476593785*sx;
    *wwy++ += -650877.476593785*sy;
    wx[ 0] = wx[10] + wx[15];
    wx[ 1] = wx[11] + wx[16];
    wx[ 2] = wx[12] + wx[17];
    wx[ 3] = wx[13] + wx[18];
    wx[ 4] = wx[14] + wx[19];
    wx[ 5] = wx[14] - wx[19];
    wx[ 6] = wx[13] - wx[18];
    wx[ 7] = wx[12] - wx[17];
    wx[ 8] = wx[11] - wx[16];
    wx[ 9] = wx[10] - wx[15];
    wy[ 0] = wy[10] + wy[15];
    wy[ 1] = wy[11] + wy[16];
    wy[ 2] = wy[12] + wy[17];
    wy[ 3] = wy[13] + wy[18];
    wy[ 4] = wy[14] + wy[19];
    wy[ 5] = wy[14] - wy[19];
    wy[ 6] = wy[13] - wy[18];
    wy[ 7] = wy[12] - wy[17];
    wy[ 8] = wy[11] - wy[16];
    wy[ 9] = wy[10] - wy[15];
    /* end loop unrolled code generated by perl */

    /* and the outputs */
    for(ilayer=0;ilayer<nlayer;ilayer++) {
      out = 0.;
      temp = (char*)PyArray_GETPTR3(infunc_,ilayer,yi-4,xi-4); /* set temp to point to corner of interpolation region */
      for(i=0;i<10;i++) {
        interp_vstrip = 0.;
        for(j=0;j<10;j++) {
          interp_vstrip += wx[j]*(*(double*)temp);
          temp += ds;
        }
        out += interp_vstrip*wy[i];
        temp += ds2; /* jump to next row of input image */
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
  long i_in, iy, ix, ipos, ds;
  double **wx_ar, **wy_ar;
  double x, y, cx, sx, cy, sy;
  double xfh, yfh, *wx, *wy, *wwx, *wwy;
  long *xi, *yi; /* frac and integer parts of abscissae; note 'xfh' and 'yfh' will have 1/2 subtracted */
  char *temp;
  long ngy,ngx,npi,nxo,nyo;
  double out, interp_vstrip;

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

  ds = infunc_->strides[1];

  /* allocate arrays */
  wx_ar = (double**)malloc((size_t)(nxo*sizeof(double*)));
  for(ix=0;ix<nxo;ix++) wx_ar[ix] = (double*)malloc((size_t)(21*sizeof(double)));
  wy_ar = (double**)malloc((size_t)(nyo*sizeof(double*)));
  for(iy=0;iy<nyo;iy++) wy_ar[iy] = (double*)malloc((size_t)(21*sizeof(double)));
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
      wwx=wx+10;
      cx = cos(7.795042160878816462e-02*xfh);
      *wwx++ = 1912.40267850101*cx;
      *wwx++ = -12176.9938608718*cx;
      *wwx++ = 32463.3012682714*cx;
      *wwx++ = -43429.0459588089*cx;
      *wwx++ = 21230.9659767686*cx;
      sx = sin(7.795042160878816462e-02*xfh);
      *wwx++ = -49042.3061911076*sx;
      *wwx++ = 410355.593860622*sx;
      *wwx++ = -1555126.53918249*sx;
      *wwx++ = 3501335.76171435*sx;
      *wwx++ = -5159499.1491332*sx;
      wwx=wx+10;
      cx = cos(2.269252977160159945e-01*xfh);
      *wwx++ += -4927.00410046915*cx;
      *wwx++ += 31594.8125350013*cx;
      *wwx++ += -84620.6445366486*cx;
      *wwx++ += 113528.194562436*cx;
      *wwx++ += -55575.5507391098*cx;
      sx = sin(2.269252977160159945e-01*xfh);
      *wwx++ += 43237.5141237444*sx;
      *wwx++ += -363739.086758674*sx;
      *wwx++ += 1383601.73837127*sx;
      *wwx++ += -3122480.58932711*sx;
      *wwx++ += 4606470.74368757*sx;
      wwx=wx+10;
      cx = cos(3.557380180911379752e-01*xfh);
      *wwx++ += 5835.90561316373*cx;
      /* interpolation weights */
      *wwx++ += -37854.7594904635*cx;
      *wwx++ += 102189.161922347*cx;
      *wwx++ += -137779.981708223*cx;
      *wwx++ += 67609.7669203679*cx;
      sx = sin(3.557380180911379752e-01*xfh);
      *wwx++ += -32463.3907553235*sx;
      *wwx++ += 275501.443053076*sx;
      *wwx++ += -1054523.73112156*sx;
      *wwx++ += 2389400.02886295*sx;
      *wwx++ += -3531921.53431643*sx;
      wwx=wx+10;
      cx = cos(4.529461196132943956e-01*xfh);
      *wwx++ += -4322.72244949997*cx;
      *wwx++ += 28369.9538060603*cx;
      *wwx++ += -77242.1292497577*cx;
      *wwx++ += 104725.479751602*cx;
      *wwx++ += -51530.6247479867*cx;
      sx = sin(4.529461196132943956e-01*xfh);
      *wwx++ += 18759.6895246111*sx;
      *wwx++ += -160638.901862404*sx;
      *wwx++ += 618972.720736964*sx;
      *wwx++ += -1408673.16724246*sx;
      *wwx++ += 2086791.19148948*sx;
      wwx=wx+10;
      cx = cos(5.099362658787808256e-01*xfh);
      *wwx++ += 1501.41887706351*cx;
      *wwx++ += -9933.01974301992*cx;
      *wwx++ += 27210.3468013967*cx;
      *wwx++ += -37044.7834316057*cx;
      *wwx++ += 18266.0493034857*cx;
      sx = sin(5.099362658787808256e-01*xfh);
      *wwx++ += -5760.49192550392*sx;
      *wwx++ += 49630.9882615042*sx;
      *wwx++ += -192138.896175493*sx;
      *wwx++ += 438666.473889707*sx;
      *wwx++ += -650877.476593785*sx;
      wx[ 0] = wx[10] + wx[15];
      wx[ 1] = wx[11] + wx[16];
      wx[ 2] = wx[12] + wx[17];
      wx[ 3] = wx[13] + wx[18];
      wx[ 4] = wx[14] + wx[19];
      wx[ 5] = wx[14] - wx[19];
      wx[ 6] = wx[13] - wx[18];
      wx[ 7] = wx[12] - wx[17];
      wx[ 8] = wx[11] - wx[16];
      wx[ 9] = wx[10] - wx[15];
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
      wwy=wy+10;
      cy = cos(7.795042160878816462e-02*yfh);
      *wwy++ = 1912.40267850101*cy;
      *wwy++ = -12176.9938608718*cy;
      *wwy++ = 32463.3012682714*cy;
      *wwy++ = -43429.0459588089*cy;
      *wwy++ = 21230.9659767686*cy;
      sy = sin(7.795042160878816462e-02*yfh);
      *wwy++ = -49042.3061911076*sy;
      *wwy++ = 410355.593860622*sy;
      *wwy++ = -1555126.53918249*sy;
      *wwy++ = 3501335.76171435*sy;
      *wwy++ = -5159499.1491332*sy;
      wwy=wy+10;
      cy = cos(2.269252977160159945e-01*yfh);
      *wwy++ += -4927.00410046915*cy;
      *wwy++ += 31594.8125350013*cy;
      *wwy++ += -84620.6445366486*cy;
      *wwy++ += 113528.194562436*cy;
      *wwy++ += -55575.5507391098*cy;
      sy = sin(2.269252977160159945e-01*yfh);
      *wwy++ += 43237.5141237444*sy;
      *wwy++ += -363739.086758674*sy;
      *wwy++ += 1383601.73837127*sy;
      *wwy++ += -3122480.58932711*sy;
      *wwy++ += 4606470.74368757*sy;
      wwy=wy+10;
      cy = cos(3.557380180911379752e-01*yfh);
      *wwy++ += 5835.90561316373*cy;
      *wwy++ += -37854.7594904635*cy;
      *wwy++ += 102189.161922347*cy;
      *wwy++ += -137779.981708223*cy;
      *wwy++ += 67609.7669203679*cy;
      sy = sin(3.557380180911379752e-01*yfh);
      *wwy++ += -32463.3907553235*sy;
      *wwy++ += 275501.443053076*sy;
      *wwy++ += -1054523.73112156*sy;
      *wwy++ += 2389400.02886295*sy;
      *wwy++ += -3531921.53431643*sy;
      wwy=wy+10;
      cy = cos(4.529461196132943956e-01*yfh);
      *wwy++ += -4322.72244949997*cy;
      *wwy++ += 28369.9538060603*cy;
      *wwy++ += -77242.1292497577*cy;
      *wwy++ += 104725.479751602*cy;
      *wwy++ += -51530.6247479867*cy;
      sy = sin(4.529461196132943956e-01*yfh);
      *wwy++ += 18759.6895246111*sy;
      *wwy++ += -160638.901862404*sy;
      *wwy++ += 618972.720736964*sy;
      *wwy++ += -1408673.16724246*sy;
      *wwy++ += 2086791.19148948*sy;
      wwy=wy+10;
      cy = cos(5.099362658787808256e-01*yfh);
      *wwy++ += 1501.41887706351*cy;
      *wwy++ += -9933.01974301992*cy;
      *wwy++ += 27210.3468013967*cy;
      *wwy++ += -37044.7834316057*cy;
      *wwy++ += 18266.0493034857*cy;
      sy = sin(5.099362658787808256e-01*yfh);
      *wwy++ += -5760.49192550392*sy;
      *wwy++ += 49630.9882615042*sy;
      *wwy++ += -192138.896175493*sy;
      *wwy++ += 438666.473889707*sy;
      *wwy++ += -650877.476593785*sy;
      wy[ 0] = wy[10] + wy[15];
      wy[ 1] = wy[11] + wy[16];
      wy[ 2] = wy[12] + wy[17];
      wy[ 3] = wy[13] + wy[18];
      wy[ 4] = wy[14] + wy[19];
      wy[ 5] = wy[14] - wy[19];
      wy[ 6] = wy[13] - wy[18];
      wy[ 7] = wy[12] - wy[17];
      wy[ 8] = wy[11] - wy[16];
      wy[ 9] = wy[10] - wy[15];
    }

    /* ... and now we can do the interpolation */
    ipos=0;
    for(iy=0;iy<nyo;iy++) { /* output pixel row */
      wy = wy_ar[iy];
      for(ix=0;ix<nxo;ix++) { /* output pixel column */
        wx = wx_ar[ix];
        out = 0.;
        for(i=0;i<10;i++) {
          interp_vstrip = 0.;
          temp = PyArray_GETPTR2(infunc_,yi[iy]-4+i,xi[ix]-4); /* set temp to beginning of row of interpolation region */
          for(j=0;j<10;j++) {
            interp_vstrip += wx[j]*(*(double*)temp);
            temp += ds;
          }
          out += interp_vstrip*wy[i];
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

  return(Py_None);
  /* -- this is the end of the function if it executed normally -- */
}

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

