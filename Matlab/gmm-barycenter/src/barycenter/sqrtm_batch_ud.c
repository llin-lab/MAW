#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
#define lapack_int int
// ported function from lapacke
inline lapack_int LAPACKE_dsyev_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, double* a, lapack_int lda,
                                double* w, double* work, lapack_int lwork ) {
	lapack_int info = 0;
	assert(matrix_layout == LAPACK_COL_MAJOR);
    	dsyev_( &jobz, &uplo, &n, a, &lda, w, work, &lwork, &info );
	if (info < 0) { info = info - 1; }
	return info;
}
#else
#include "cblas.h"
#include "lapacke.h"
#endif

/*
 D = sqrtm_batch_ud(V, m);
 
 V is a d x d x n matrices
 m is a d x d symmetric matrix
 D(i) = trace(V(:,:,i)) - 2 * trace(sqrtm(m * V(:,:,i) * m))
*/


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    const mwSize *dimensions = mxGetDimensions(prhs[0]);;
    double *V, *m, *D;

    V = mxGetPr(prhs[0]);
    m = mxGetPr(prhs[1]);
    
    mwSize d, n;
    d=dimensions[0];
    n=dimensions[2];
    
    plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
    D = mxGetPr(plhs[0]);
    
    double *Vtmp1, *Vtmp2, *L, *Q, *work;
    Vtmp1 = (double*) malloc(d*d*n*sizeof(double));
    Vtmp2 = (double*) malloc(d*d*sizeof(double));
    Q = (double*) malloc(d*d*sizeof(double));
    lapack_int lwork = 26*d, suppz; 
    int info;
    L = (double*) malloc(d*3*sizeof(double));
    work = (double*) malloc(lwork*sizeof(double));
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d*n, d, 1.0, m, d, V, d, 0.0, Vtmp1, d);
    mwSize i,j,k;
    for (i=0; i<n; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0, Vtmp1+d*d*i, d, m, d, 0.0, Vtmp2, d);
        memcpy(Q, Vtmp2, sizeof(double)*d*d);
        info=LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V',
                                 'L', d, Vtmp2, 
                                 d, L, 
                                 work, lwork);
        assert(info == 0);
        for (j=0; j<d; ++j) {
            double lambda = sqrt(fabs(L[j]));
            for (k=0; k<d; ++k) {
                Q[j*d+k] = Vtmp2[j*d+k] * lambda;
            }
        }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d, d, d, 1., Q, d, Vtmp2, d, 0.0, Vtmp1 + d*d*i, d);
        D[i] = 0;
        for (j=0; j<d*d; j+=(d+1)) {
            D[i] += V[d*d*i+j] - 2*Vtmp1[d*d*i+j];
        }
    }
    
    free(Vtmp1); free(Vtmp2); free(Q); free(L); free(work);
}
