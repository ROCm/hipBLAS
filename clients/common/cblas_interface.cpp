/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#include "cblas_interface.h"
#include "cblas.h"
#include "hipblas.h"
#include "utility.h"
#include <cmath>
#include <memory>
#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
*/

#ifdef __cplusplus
extern "C" {
#endif

void strtri_(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri_(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
//  void    ctrtri_(char* uplo, char* diag, int* n, hipComplex* A,  int* lda, int *info);
//  void    ztrtri_(char* uplo, char* diag, int* n, hipDoubleComplex* A, int* lda, int *info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
//  void    cgetrf_(int* m, int* n, hipComplex* A, int* lda, int* ipiv, int *info);
//  void    zgetrf_(int* m, int* n, hipDoubleComplex* A, int* lda, int* ipiv, int *info);

void spotrf_(char* uplo, int* m, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* m, double* A, int* lda, int* info);

#ifdef __cplusplus
}
#endif

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// axpy
template <>
void cblas_axpy<hipblasHalf>(
    int n, const hipblasHalf alpha, const hipblasHalf* x, int incx, hipblasHalf* y, int incy)
{
    size_t        abs_incx = incx >= 0 ? incx : -incx;
    size_t        abs_incy = incy >= 0 ? incy : -incy;
    vector<float> x_float(n * abs_incx);
    vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }

    cblas_saxpy(n, half_to_float(alpha), x_float.data(), incx, y_float.data(), incy);

    for(size_t i = 0; i < n; i++)
    {
        y[i * abs_incx] = float_to_half(y_float[i * abs_incx]);
    }
}

template <>
void cblas_axpy<float>(int n, const float alpha, const float* x, int incx, float* y, int incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<double>(int n, const double alpha, const double* x, int incx, double* y, int incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<hipComplex>(
    int n, const hipComplex alpha, const hipComplex* x, int incx, hipComplex* y, int incy)
{
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<hipDoubleComplex>(int                     n,
                                  const hipDoubleComplex  alpha,
                                  const hipDoubleComplex* x,
                                  int                     incx,
                                  hipDoubleComplex*       y,
                                  int                     incy)
{
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// scal
template <>
void cblas_scal<float>(int n, const float alpha, float* x, int incx)
{
    cblas_sscal(n, alpha, x, incx);
}

template <>
void cblas_scal<double>(int n, const double alpha, double* x, int incx)
{
    cblas_dscal(n, alpha, x, incx);
}

template <>
void cblas_scal<hipComplex>(int n, const hipComplex alpha, hipComplex* x, int incx)
{
    cblas_cscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<hipComplex, float>(int n, const float alpha, hipComplex* x, int incx)
{
    cblas_csscal(n, alpha, x, incx);
}

template <>
void cblas_scal<hipDoubleComplex>(int                    n,
                                  const hipDoubleComplex alpha,
                                  hipDoubleComplex*      x,
                                  int                    incx)
{
    cblas_zscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<hipDoubleComplex, double>(int n, const double alpha, hipDoubleComplex* x, int incx)
{
    cblas_zdscal(n, alpha, x, incx);
}

// copy
template <>
void cblas_copy<float>(int n, float* x, int incx, float* y, int incy)
{
    cblas_scopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<double>(int n, double* x, int incx, double* y, int incy)
{
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<hipComplex>(int n, hipComplex* x, int incx, hipComplex* y, int incy)
{
    cblas_ccopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<hipDoubleComplex>(
    int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
{
    cblas_zcopy(n, x, incx, y, incy);
}

// swap
template <>
void cblas_swap<float>(int n, float* x, int incx, float* y, int incy)
{
    cblas_sswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<double>(int n, double* x, int incx, double* y, int incy)
{
    cblas_dswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<hipComplex>(int n, hipComplex* x, int incx, hipComplex* y, int incy)
{
    cblas_cswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<hipDoubleComplex>(
    int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
{
    cblas_zswap(n, x, incx, y, incy);
}

// dot
template <>
void cblas_dot<hipblasHalf>(
    int n, const hipblasHalf* x, int incx, const hipblasHalf* y, int incy, hipblasHalf* result)
{
    size_t        abs_incx = incx >= 0 ? incx : -incx;
    size_t        abs_incy = incy >= 0 ? incy : -incy;
    vector<float> x_float(n * abs_incx);
    vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = half_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = half_to_float(y[i * abs_incy]);
    }
    *result = float_to_half(cblas_sdot(n, x_float.data(), incx, y_float.data(), incy));
}

template <>
void cblas_dot<hipblasBfloat16>(int                    n,
                                const hipblasBfloat16* x,
                                int                    incx,
                                const hipblasBfloat16* y,
                                int                    incy,
                                hipblasBfloat16*       result)
{
    size_t        abs_incx = incx >= 0 ? incx : -incx;
    size_t        abs_incy = incy >= 0 ? incy : -incy;
    vector<float> x_float(n * abs_incx);
    vector<float> y_float(n * abs_incy);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = bfloat16_to_float(x[i * abs_incx]);
        y_float[i * abs_incy] = bfloat16_to_float(y[i * abs_incy]);
    }
    *result = float_to_bfloat16(cblas_sdot(n, x_float.data(), incx, y_float.data(), incy));
}

template <>
void cblas_dot<float>(int n, const float* x, int incx, const float* y, int incy, float* result)
{
    *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
void cblas_dot<double>(int n, const double* x, int incx, const double* y, int incy, double* result)
{
    *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
void cblas_dot<hipComplex>(
    int n, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* result)
{
    cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dot<hipDoubleComplex>(int                     n,
                                 const hipDoubleComplex* x,
                                 int                     incx,
                                 const hipDoubleComplex* y,
                                 int                     incy,
                                 hipDoubleComplex*       result)
{
    cblas_zdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<hipComplex>(
    int n, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* result)
{
    cblas_cdotc_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<hipDoubleComplex>(int                     n,
                                  const hipDoubleComplex* x,
                                  int                     incx,
                                  const hipDoubleComplex* y,
                                  int                     incy,
                                  hipDoubleComplex*       result)
{
    cblas_zdotc_sub(n, x, incx, y, incy, result);
}

// nrm2
template <>
void cblas_nrm2<float, float>(int n, const float* x, int incx, float* result)
{
    *result = cblas_snrm2(n, x, incx);
}

template <>
void cblas_nrm2<double, double>(int n, const double* x, int incx, double* result)
{
    *result = cblas_dnrm2(n, x, incx);
}

template <>
void cblas_nrm2<hipComplex, float>(int n, const hipComplex* x, int incx, float* result)
{
    *result = cblas_scnrm2(n, x, incx);
}

template <>
void cblas_nrm2<hipDoubleComplex, double>(int                     n,
                                          const hipDoubleComplex* x,
                                          int                     incx,
                                          double*                 result)
{
    *result = cblas_dznrm2(n, x, incx);
}

///////////////////
// rot functions //
///////////////////
// LAPACK fortran library functionality
extern "C" {
void crot_(const int*        n,
           hipComplex*       cx,
           const int*        incx,
           hipComplex*       cy,
           const int*        incy,
           const float*      c,
           const hipComplex* s);
void csrot_(const int*   n,
            hipComplex*  cx,
            const int*   incx,
            hipComplex*  cy,
            const int*   incy,
            const float* c,
            const float* s);

void crotg_(hipComplex* a, hipComplex* b, float* c, hipComplex* s);
}

// rot
template <>
void cblas_rot<float>(int n, float* x, int incx, float* y, int incy, float c, float s)
{
    cblas_srot(n, x, incx, y, incy, c, s);
}

template <>
void cblas_rot<double>(int n, double* x, int incx, double* y, int incy, double c, double s)
{
    cblas_drot(n, x, incx, y, incy, c, s);
}

template <>
void cblas_rot<hipComplex, float>(
    int n, hipComplex* x, int incx, hipComplex* y, int incy, float c, hipComplex s)
{
    crot_(&n, x, &incx, y, &incx, &c, &s);
}

template <>
void cblas_rot<hipComplex, float, float>(
    int n, hipComplex* x, int incx, hipComplex* y, int incy, float c, float s)
{
    csrot_(&n, x, &incx, y, &incx, &c, &s);
}

// rotg
template <>
void cblas_rotg<float>(float* a, float* b, float* c, float* s)
{
    cblas_srotg(a, b, c, s);
}

template <>
void cblas_rotg<double>(double* a, double* b, double* c, double* s)
{
    cblas_drotg(a, b, c, s);
}

template <>
void cblas_rotg<hipComplex, float>(hipComplex* a, hipComplex* b, float* c, hipComplex* s)
{
    crotg_(a, b, c, s);
}

// rotm
template <>
void cblas_rotm<float>(int n, float* x, int incx, float* y, int incy, float* param)
{
    cblas_srotm(n, x, incx, y, incy, param);
}

template <>
void cblas_rotm<double>(int n, double* x, int incx, double* y, int incy, double* param)
{
    cblas_drotm(n, x, incx, y, incy, param);
}

// rotmg
template <>
void cblas_rotmg<float>(float* d1, float* d2, float* x1, float* y1, float* param)
{
    cblas_srotmg(d1, d2, x1, *y1, param);
}

template <>
void cblas_rotmg<double>(double* d1, double* d2, double* x1, double* y1, double* param)
{
    cblas_drotmg(d1, d2, x1, *y1, param);
}

// asum
template <>
void cblas_asum<float, float>(int n, const float* x, int incx, float* result)
{
    *result = cblas_sasum(n, x, incx);
}

template <>
void cblas_asum<double, double>(int n, const double* x, int incx, double* result)
{
    *result = cblas_dasum(n, x, incx);
}

template <>
void cblas_asum<hipComplex, float>(int n, const hipComplex* x, int incx, float* result)
{
    *result = cblas_scasum(n, x, incx);
}

template <>
void cblas_asum<hipDoubleComplex, double>(int                     n,
                                          const hipDoubleComplex* x,
                                          int                     incx,
                                          double*                 result)
{
    *result = cblas_dzasum(n, x, incx);
}

// amax
template <>
void cblas_iamax<float>(int n, const float* x, int incx, int* result)
{
    *result = (int)cblas_isamax(n, x, incx);
}

template <>
void cblas_iamax<double>(int n, const double* x, int incx, int* result)
{
    *result = (int)cblas_idamax(n, x, incx);
}

template <>
void cblas_iamax<hipComplex>(int n, const hipComplex* x, int incx, int* result)
{
    *result = (int)cblas_icamax(n, x, incx);
}

template <>
void cblas_iamax<hipDoubleComplex>(int n, const hipDoubleComplex* x, int incx, int* result)
{
    *result = (int)cblas_izamax(n, x, incx);
}

// amin
// amin is not implemented in cblas, make local version
template <typename T>
double abs_helper(T val)
{
    return val < 0 ? -val : val;
}

template <>
double abs_helper(hipComplex val)
{
    return std::abs(val.x) + std::abs(val.y);
}

template <>
double abs_helper(hipDoubleComplex val)
{
    return std::abs(val.x) + std::abs(val.y);
}

template <typename T>
int cblas_iamin_helper(int N, const T* X, int incx)
{
    int minpos = -1;
    if(N > 0 && incx > 0)
    {
        auto min = abs_helper(X[0]);
        minpos   = 0;
        for(size_t i = 1; i < N; ++i)
        {
            auto a = abs_helper(X[i * incx]);
            if(a < min)
            {
                min    = a;
                minpos = i;
            }
        }
    }
    return minpos;
}

template <>
void cblas_iamin<float>(int n, const float* x, int incx, int* result)
{
    *result = (int)cblas_iamin_helper(n, x, incx);
}

template <>
void cblas_iamin<double>(int n, const double* x, int incx, int* result)
{
    *result = (int)cblas_iamin_helper(n, x, incx);
}

template <>
void cblas_iamin<hipComplex>(int n, const hipComplex* x, int incx, int* result)
{
    *result = (int)cblas_iamin_helper(n, x, incx);
}

template <>
void cblas_iamin<hipDoubleComplex>(int n, const hipDoubleComplex* x, int incx, int* result)
{
    *result = (int)cblas_iamin_helper(n, x, incx);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gemv
template <>
void cblas_gemv<float>(hipblasOperation_t transA,
                       int                m,
                       int                n,
                       float              alpha,
                       float*             A,
                       int                lda,
                       float*             x,
                       int                incx,
                       float              beta,
                       float*             y,
                       int                incy)
{
    cblas_sgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_gemv<double>(hipblasOperation_t transA,
                        int                m,
                        int                n,
                        double             alpha,
                        double*            A,
                        int                lda,
                        double*            x,
                        int                incx,
                        double             beta,
                        double*            y,
                        int                incy)
{
    cblas_dgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_gemv<hipComplex>(hipblasOperation_t transA,
                            int                m,
                            int                n,
                            hipComplex         alpha,
                            hipComplex*        A,
                            int                lda,
                            hipComplex*        x,
                            int                incx,
                            hipComplex         beta,
                            hipComplex*        y,
                            int                incy)
{
    cblas_cgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_gemv<hipDoubleComplex>(hipblasOperation_t transA,
                                  int                m,
                                  int                n,
                                  hipDoubleComplex   alpha,
                                  hipDoubleComplex*  A,
                                  int                lda,
                                  hipDoubleComplex*  x,
                                  int                incx,
                                  hipDoubleComplex   beta,
                                  hipDoubleComplex*  y,
                                  int                incy)
{
    cblas_zgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// symv
template <>
void cblas_symv<float>(hipblasFillMode_t uplo,
                       int               n,
                       float             alpha,
                       float*            A,
                       int               lda,
                       float*            x,
                       int               incx,
                       float             beta,
                       float*            y,
                       int               incy)
{
    cblas_ssymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_symv<double>(hipblasFillMode_t uplo,
                        int               n,
                        double            alpha,
                        double*           A,
                        int               lda,
                        double*           x,
                        int               incx,
                        double            beta,
                        double*           y,
                        int               incy)
{
    cblas_dsymv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

//  template<>
//  void cblas_hemv<hipComplex>(hipblasFillMode_t uplo, int n,
//                          hipComplex alpha,
//                          hipComplex *A, int lda,
//                          hipComplex *x, int incx,
//                          hipComplex beta, hipComplex *y, int incy)
//  {
//      cblas_chemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
//  }

//  template<>
//  void cblas_hemv<hipDoubleComplex>(hipblasFillMode_t uplo, int n,
//                          hipDoubleComplex alpha,
//                          hipDoubleComplex *A, int lda,
//                          hipDoubleComplex *x, int incx,
//                          hipDoubleComplex beta, hipDoubleComplex *y, int incy)
//  {
//      cblas_zhemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
//  }

template <>
void cblas_ger<float>(
    int m, int n, float alpha, float* x, int incx, float* y, int incy, float* A, int lda)
{
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<double>(
    int m, int n, double alpha, double* x, int incx, double* y, int incy, double* A, int lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

// syr
template <>
void cblas_syr<float>(
    hipblasFillMode_t uplo, int n, float alpha, float* x, int incx, float* A, int lda)
{
    cblas_ssyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void cblas_syr<double>(
    hipblasFillMode_t uplo, int n, double alpha, double* x, int incx, double* A, int lda)
{
    cblas_dsyr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

// trmv
template <>
void cblas_trmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const float*       A,
                int                lda,
                float*             x,
                int                incx)
{
    cblas_strmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_trmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const double*      A,
                int                lda,
                double*            x,
                int                incx)
{
    cblas_dtrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// potrf
template <>
int cblas_potrf(char uplo, int m, float* A, int lda)
{
    int info;
    spotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
int cblas_potrf(char uplo, int m, double* A, int lda)
{
    int info;
    dpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

// trsv
template <>
void cblas_trsv<float>(hipblasHandle_t    handle,
                       hipblasFillMode_t  uplo,
                       hipblasOperation_t transA,
                       hipblasDiagType_t  diag,
                       int                m,
                       const float*       A,
                       int                lda,
                       float*             x,
                       int                incx)
{
    cblas_strsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_trsv<double>(hipblasHandle_t    handle,
                        hipblasFillMode_t  uplo,
                        hipblasOperation_t transA,
                        hipblasDiagType_t  diag,
                        int                m,
                        const double*      A,
                        int                lda,
                        double*            x,
                        int                incx)
{
    cblas_dtrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// gemm

template <>
void cblas_gemm<hipblasHalf>(hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int                m,
                             int                n,
                             int                k,
                             hipblasHalf        alpha,
                             hipblasHalf*       A,
                             int                lda,
                             hipblasHalf*       B,
                             int                ldb,
                             hipblasHalf        beta,
                             hipblasHalf*       C,
                             int                ldc)
{
    // cblas does not support hipblasHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing
    float alpha_float = half_to_float(alpha);
    float beta_float  = half_to_float(beta);

    int sizeA = transA == HIPBLAS_OP_N ? k * lda : m * lda;
    int sizeB = transB == HIPBLAS_OP_N ? n * ldb : k * ldb;
    int sizeC = n * ldc;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());
    std::unique_ptr<float[]> C_float(new float[sizeC]());

    for(int i = 0; i < sizeA; i++)
    {
        A_float[i] = half_to_float(A[i]);
    }
    for(int i = 0; i < sizeB; i++)
    {
        B_float[i] = half_to_float(B[i]);
    }
    for(int i = 0; i < sizeC; i++)
    {
        C_float[i] = half_to_float(C[i]);
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha_float,
                const_cast<const float*>(A_float.get()),
                lda,
                const_cast<const float*>(B_float.get()),
                ldb,
                beta_float,
                static_cast<float*>(C_float.get()),
                ldc);

    for(int i = 0; i < sizeC; i++)
    {
        C[i] = float_to_half(C_float[i]);
    }
}

template <>
void cblas_gemm<float>(hipblasOperation_t transA,
                       hipblasOperation_t transB,
                       int                m,
                       int                n,
                       int                k,
                       float              alpha,
                       float*             A,
                       int                lda,
                       float*             B,
                       int                ldb,
                       float              beta,
                       float*             C,
                       int                ldc)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<double>(hipblasOperation_t transA,
                        hipblasOperation_t transB,
                        int                m,
                        int                n,
                        int                k,
                        double             alpha,
                        double*            A,
                        int                lda,
                        double*            B,
                        int                ldb,
                        double             beta,
                        double*            C,
                        int                ldc)
{
    cblas_dgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipComplex>(hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int                m,
                            int                n,
                            int                k,
                            hipComplex         alpha,
                            hipComplex*        A,
                            int                lda,
                            hipComplex*        B,
                            int                ldb,
                            hipComplex         beta,
                            hipComplex*        C,
                            int                ldc)
{
    //just directly cast, since transA, transB are integers in the enum
    cblas_cgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipDoubleComplex>(hipblasOperation_t transA,
                                  hipblasOperation_t transB,
                                  int                m,
                                  int                n,
                                  int                k,
                                  hipDoubleComplex   alpha,
                                  hipDoubleComplex*  A,
                                  int                lda,
                                  hipDoubleComplex*  B,
                                  int                ldb,
                                  hipDoubleComplex   beta,
                                  hipDoubleComplex*  C,
                                  int                ldc)
{
    cblas_zgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// trsm
template <>
void cblas_trsm<float>(hipblasSideMode_t  side,
                       hipblasFillMode_t  uplo,
                       hipblasOperation_t transA,
                       hipblasDiagType_t  diag,
                       int                m,
                       int                n,
                       float              alpha,
                       const float*       A,
                       int                lda,
                       float*             B,
                       int                ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trsm<double>(hipblasSideMode_t  side,
                        hipblasFillMode_t  uplo,
                        hipblasOperation_t transA,
                        hipblasDiagType_t  diag,
                        int                m,
                        int                n,
                        double             alpha,
                        const double*      A,
                        int                lda,
                        double*            B,
                        int                ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

//  template<>
//  void cblas_trsm<hipComplex>( hipblasSideMode_t side, hipblasFillMode_t uplo,
//                          hipblasOperation_t transA, hipblasDiagType_t diag,
//                          int m, int n,
//                          hipComplex alpha,
//                          const hipComplex *A, int lda,
//                          hipComplex *B, int ldb)
//  {
//      //just directly cast, since transA, transB are integers in the enum
//      cblas_ctrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA,
//      (CBLAS_DIAG)diag, m, n, &alpha, A, lda, B, ldb);
//  }

//  template<>
//  void cblas_trsm<hipDoubleComplex>( hipblasSideMode_t side, hipblasFillMode_t uplo,
//                          hipblasOperation_t transA, hipblasDiagType_t diag,
//                          int m, int n,
//                          hipDoubleComplex alpha,
//                          const hipDoubleComplex *A, int lda,
//                          hipDoubleComplex *B, int ldb)
//  {
//      //just directly cast, since transA, transB are integers in the enum
//      cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA,
//      (CBLAS_DIAG)diag, m, n, &alpha, A, lda, B, ldb);
//  }

// trtri
template <>
int cblas_trtri<float>(char uplo, char diag, int n, float* A, int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    int info;
    strtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

template <>
int cblas_trtri<double>(char uplo, char diag, int n, double* A, int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    int info;
    dtrtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

// trmm
template <>
void cblas_trmm<float>(hipblasSideMode_t  side,
                       hipblasFillMode_t  uplo,
                       hipblasOperation_t transA,
                       hipblasDiagType_t  diag,
                       int                m,
                       int                n,
                       float              alpha,
                       const float*       A,
                       int                lda,
                       float*             B,
                       int                ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_strmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trmm<double>(hipblasSideMode_t  side,
                        hipblasFillMode_t  uplo,
                        hipblasOperation_t transA,
                        hipblasDiagType_t  diag,
                        int                m,
                        int                n,
                        double             alpha,
                        const double*      A,
                        int                lda,
                        double*            B,
                        int                ldb)
{
    // just directly cast, since transA, transB are integers in the enum
    cblas_dtrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
}

//  template<>
//  void cblas_trmm<hipComplex>( hipblasSideMode_t side, hipblasFillMode_t uplo,
//                          hipblasOperation_t transA, hipblasDiagType_t diag,
//                          int m, int n,
//                          hipComplex alpha,
//                          const hipComplex *A, int lda,
//                          hipComplex *B, int ldb)
//  {
//      //just directly cast, since transA, transB are integers in the enum
//      cblas_ctrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA,
//      (CBLAS_DIAG)diag, m, n, &alpha, A, lda, B, ldb);
//  }

//  template<>
//  void cblas_trmm<hipDoubleComplex>( hipblasSideMode_t side, hipblasFillMode_t uplo,
//                          hipblasOperation_t transA, hipblasDiagType_t diag,
//                          int m, int n,
//                          hipDoubleComplex alpha,
//                          const hipDoubleComplex *A, int lda,
//                          hipDoubleComplex *B, int ldb)
//  {
//      //just directly cast, since transA, transB are integers in the enum
//      cblas_ztrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transA,
//      (CBLAS_DIAG)diag, m, n, &alpha, A, lda, B, ldb);
//  }

// getrf
template <>
int cblas_getrf<float>(int m, int n, float* A, int lda, int* ipiv)
{
    int info;
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
int cblas_getrf<double>(int m, int n, double* A, int lda, int* ipiv)
{
    int info;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// template<>
// int cblas_getrf<hipComplex>(int m,
//                         int n,
//                         hipComplex *A, int lda,
//                         int *ipiv)
// {
//     int info;
//     cgetrf_(&m, &n, A, &lda, ipiv, &info);
//     return info;
// }

// template<>
// int cblas_getrf<hipDoubleComplex>(int m,
//                         int n,
//                         hipDoubleComplex *A, int lda,
//                         int *ipiv)
// {
//     int info;
//     zgetrf_(&m, &n, A, &lda, ipiv, &info);
//     return info;
// }
