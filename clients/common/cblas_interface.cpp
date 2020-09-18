/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
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
void ctrtri_(char* uplo, char* diag, int* n, hipblasComplex* A, int* lda, int* info);
void ztrtri_(char* uplo, char* diag, int* n, hipblasDoubleComplex* A, int* lda, int* info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, hipblasComplex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, hipblasDoubleComplex* A, int* lda, int* ipiv, int* info);

void sgetrs_(
    char* trans, int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgetrs_(
    char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgetrs_(char*           trans,
             int*            n,
             int*            nrhs,
             hipblasComplex* A,
             int*            lda,
             int*            ipiv,
             hipblasComplex* B,
             int*            ldb,
             int*            info);
void zgetrs_(char*                 trans,
             int*                  n,
             int*                  nrhs,
             hipblasDoubleComplex* A,
             int*                  lda,
             int*                  ipiv,
             hipblasDoubleComplex* B,
             int*                  ldb,
             int*                  info);

void sgetri_(int* n, float* A, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
void cgetri_(
    int* n, hipblasComplex* A, int* lda, int* ipiv, hipblasComplex* work, int* lwork, int* info);
void zgetri_(int*                  n,
             hipblasDoubleComplex* A,
             int*                  lda,
             int*                  ipiv,
             hipblasDoubleComplex* work,
             int*                  lwork,
             int*                  info);

void sgeqrf_(int* m, int* n, float* A, int* lda, float* tau, float* work, int* lwork, int* info);
void dgeqrf_(int* m, int* n, double* A, int* lda, double* tau, double* work, int* lwork, int* info);
void cgeqrf_(int*            m,
             int*            n,
             hipblasComplex* A,
             int*            lda,
             hipblasComplex* tau,
             hipblasComplex* work,
             int*            lwork,
             int*            info);
void zgeqrf_(int*                  m,
             int*                  n,
             hipblasDoubleComplex* A,
             int*                  lda,
             hipblasDoubleComplex* tau,
             hipblasDoubleComplex* work,
             int*                  lwork,
             int*                  info);

void spotrf_(char* uplo, int* m, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* m, double* A, int* lda, int* info);
void cpotrf_(char* uplo, int* m, hipblasComplex* A, int* lda, int* info);
void zpotrf_(char* uplo, int* m, hipblasDoubleComplex* A, int* lda, int* info);

void cspr_(
    char* uplo, int* n, hipblasComplex* alpha, hipblasComplex* x, int* incx, hipblasComplex* A);

void zspr_(char*                 uplo,
           int*                  n,
           hipblasDoubleComplex* alpha,
           hipblasDoubleComplex* x,
           int*                  incx,
           hipblasDoubleComplex* A);

void csyr_(char*           uplo,
           int*            n,
           hipblasComplex* alpha,
           hipblasComplex* x,
           int*            incx,
           hipblasComplex* a,
           int*            lda);
void zsyr_(char*                 uplo,
           int*                  n,
           hipblasDoubleComplex* alpha,
           hipblasDoubleComplex* x,
           int*                  incx,
           hipblasDoubleComplex* a,
           int*                  lda);

void csymv_(char*           uplo,
            int*            n,
            hipblasComplex* alpha,
            hipblasComplex* A,
            int*            lda,
            hipblasComplex* x,
            int*            incx,
            hipblasComplex* beta,
            hipblasComplex* y,
            int*            incy);

void zsymv_(char*                 uplo,
            int*                  n,
            hipblasDoubleComplex* alpha,
            hipblasDoubleComplex* A,
            int*                  lda,
            hipblasDoubleComplex* x,
            int*                  incx,
            hipblasDoubleComplex* beta,
            hipblasDoubleComplex* y,
            int*                  incy);

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
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

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
void cblas_axpy<hipblasComplex>(int                   n,
                                const hipblasComplex  alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       y,
                                int                   incy)
{
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
void cblas_axpy<hipblasDoubleComplex>(int                         n,
                                      const hipblasDoubleComplex  alpha,
                                      const hipblasDoubleComplex* x,
                                      int                         incx,
                                      hipblasDoubleComplex*       y,
                                      int                         incy)
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
void cblas_scal<hipblasComplex>(int n, const hipblasComplex alpha, hipblasComplex* x, int incx)
{
    cblas_cscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<hipblasComplex, float>(int n, const float alpha, hipblasComplex* x, int incx)
{
    cblas_csscal(n, alpha, x, incx);
}

template <>
void cblas_scal<hipblasDoubleComplex>(int                        n,
                                      const hipblasDoubleComplex alpha,
                                      hipblasDoubleComplex*      x,
                                      int                        incx)
{
    cblas_zscal(n, &alpha, x, incx);
}

template <>
void cblas_scal<hipblasDoubleComplex, double>(int                   n,
                                              const double          alpha,
                                              hipblasDoubleComplex* x,
                                              int                   incx)
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
void cblas_copy<hipblasComplex>(int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    cblas_ccopy(n, x, incx, y, incy);
}

template <>
void cblas_copy<hipblasDoubleComplex>(
    int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy)
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
void cblas_swap<hipblasComplex>(int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    cblas_cswap(n, x, incx, y, incy);
}

template <>
void cblas_swap<hipblasDoubleComplex>(
    int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy)
{
    cblas_zswap(n, x, incx, y, incy);
}

// dot
template <>
void cblas_dot<hipblasHalf>(
    int n, const hipblasHalf* x, int incx, const hipblasHalf* y, int incy, hipblasHalf* result)
{
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

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
    size_t             abs_incx = incx >= 0 ? incx : -incx;
    size_t             abs_incy = incy >= 0 ? incy : -incy;
    std::vector<float> x_float(n * abs_incx);
    std::vector<float> y_float(n * abs_incy);

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
void cblas_dot<hipblasComplex>(int                   n,
                               const hipblasComplex* x,
                               int                   incx,
                               const hipblasComplex* y,
                               int                   incy,
                               hipblasComplex*       result)
{
    cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dot<hipblasDoubleComplex>(int                         n,
                                     const hipblasDoubleComplex* x,
                                     int                         incx,
                                     const hipblasDoubleComplex* y,
                                     int                         incy,
                                     hipblasDoubleComplex*       result)
{
    cblas_zdotu_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<hipblasComplex>(int                   n,
                                const hipblasComplex* x,
                                int                   incx,
                                const hipblasComplex* y,
                                int                   incy,
                                hipblasComplex*       result)
{
    cblas_cdotc_sub(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<hipblasDoubleComplex>(int                         n,
                                      const hipblasDoubleComplex* x,
                                      int                         incx,
                                      const hipblasDoubleComplex* y,
                                      int                         incy,
                                      hipblasDoubleComplex*       result)
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
void cblas_nrm2<hipblasComplex, float>(int n, const hipblasComplex* x, int incx, float* result)
{
    *result = cblas_scnrm2(n, x, incx);
}

template <>
void cblas_nrm2<hipblasDoubleComplex, double>(int                         n,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              double*                     result)
{
    *result = cblas_dznrm2(n, x, incx);
}

///////////////////
// rot functions //
///////////////////
// LAPACK fortran library functionality
extern "C" {
void crot_(const int*            n,
           hipblasComplex*       cx,
           const int*            incx,
           hipblasComplex*       cy,
           const int*            incy,
           const float*          c,
           const hipblasComplex* s);
void csrot_(const int*      n,
            hipblasComplex* cx,
            const int*      incx,
            hipblasComplex* cy,
            const int*      incy,
            const float*    c,
            const float*    s);

void crotg_(hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);
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
void cblas_rot<hipblasComplex, float>(
    int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy, float c, hipblasComplex s)
{
    crot_(&n, x, &incx, y, &incx, &c, &s);
}

template <>
void cblas_rot<hipblasComplex, float, float>(
    int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy, float c, float s)
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
void cblas_rotg<hipblasComplex, float>(hipblasComplex* a,
                                       hipblasComplex* b,
                                       float*          c,
                                       hipblasComplex* s)
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
void cblas_asum<hipblasComplex, float>(int n, const hipblasComplex* x, int incx, float* result)
{
    *result = cblas_scasum(n, x, incx);
}

template <>
void cblas_asum<hipblasDoubleComplex, double>(int                         n,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              double*                     result)
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
void cblas_iamax<hipblasComplex>(int n, const hipblasComplex* x, int incx, int* result)
{
    *result = (int)cblas_icamax(n, x, incx);
}

template <>
void cblas_iamax<hipblasDoubleComplex>(int n, const hipblasDoubleComplex* x, int incx, int* result)
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
double abs_helper(hipblasComplex val)
{
    return std::abs(val.real()) + std::abs(val.imag());
}

template <>
double abs_helper(hipblasDoubleComplex val)
{
    return std::abs(val.real()) + std::abs(val.imag());
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
void cblas_iamin<hipblasComplex>(int n, const hipblasComplex* x, int incx, int* result)
{
    *result = (int)cblas_iamin_helper(n, x, incx);
}

template <>
void cblas_iamin<hipblasDoubleComplex>(int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    *result = (int)cblas_iamin_helper(n, x, incx);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
template <>
void cblas_gbmv<float>(hipblasOperation_t transA,
                       int                m,
                       int                n,
                       int                kl,
                       int                ku,
                       float              alpha,
                       float*             A,
                       int                lda,
                       float*             x,
                       int                incx,
                       float              beta,
                       float*             y,
                       int                incy)
{
    cblas_sgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
void cblas_gbmv<double>(hipblasOperation_t transA,
                        int                m,
                        int                n,
                        int                kl,
                        int                ku,
                        double             alpha,
                        double*            A,
                        int                lda,
                        double*            x,
                        int                incx,
                        double             beta,
                        double*            y,
                        int                incy)
{
    cblas_dgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
void cblas_gbmv<hipblasComplex>(hipblasOperation_t transA,
                                int                m,
                                int                n,
                                int                kl,
                                int                ku,
                                hipblasComplex     alpha,
                                hipblasComplex*    A,
                                int                lda,
                                hipblasComplex*    x,
                                int                incx,
                                hipblasComplex     beta,
                                hipblasComplex*    y,
                                int                incy)
{
    cblas_cgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

template <>
void cblas_gbmv<hipblasDoubleComplex>(hipblasOperation_t    transA,
                                      int                   m,
                                      int                   n,
                                      int                   kl,
                                      int                   ku,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      hipblasDoubleComplex* x,
                                      int                   incx,
                                      hipblasDoubleComplex  beta,
                                      hipblasDoubleComplex* y,
                                      int                   incy)
{
    cblas_zgbmv(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

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
void cblas_gemv<hipblasComplex>(hipblasOperation_t transA,
                                int                m,
                                int                n,
                                hipblasComplex     alpha,
                                hipblasComplex*    A,
                                int                lda,
                                hipblasComplex*    x,
                                int                incx,
                                hipblasComplex     beta,
                                hipblasComplex*    y,
                                int                incy)
{
    cblas_cgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_gemv<hipblasDoubleComplex>(hipblasOperation_t    transA,
                                      int                   m,
                                      int                   n,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      hipblasDoubleComplex* x,
                                      int                   incx,
                                      hipblasDoubleComplex  beta,
                                      hipblasDoubleComplex* y,
                                      int                   incy)
{
    cblas_zgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)transA, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// ger
template <>
void cblas_ger<float, false>(
    int m, int n, float alpha, float* x, int incx, float* y, int incy, float* A, int lda)
{
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<double, false>(
    int m, int n, double alpha, double* x, int incx, double* y, int incy, double* A, int lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<hipblasComplex, false>(int             m,
                                      int             n,
                                      hipblasComplex  alpha,
                                      hipblasComplex* x,
                                      int             incx,
                                      hipblasComplex* y,
                                      int             incy,
                                      hipblasComplex* A,
                                      int             lda)
{
    cblas_cgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<hipblasComplex, true>(int             m,
                                     int             n,
                                     hipblasComplex  alpha,
                                     hipblasComplex* x,
                                     int             incx,
                                     hipblasComplex* y,
                                     int             incy,
                                     hipblasComplex* A,
                                     int             lda)
{
    cblas_cgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<hipblasDoubleComplex, false>(int                   m,
                                            int                   n,
                                            hipblasDoubleComplex  alpha,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasDoubleComplex* y,
                                            int                   incy,
                                            hipblasDoubleComplex* A,
                                            int                   lda)
{
    cblas_zgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_ger<hipblasDoubleComplex, true>(int                   m,
                                           int                   n,
                                           hipblasDoubleComplex  alpha,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasDoubleComplex* A,
                                           int                   lda)
{
    cblas_zgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

// hbmv
template <>
void cblas_hbmv<hipblasComplex>(hipblasFillMode_t uplo,
                                int               n,
                                int               k,
                                hipblasComplex    alpha,
                                hipblasComplex*   A,
                                int               lda,
                                hipblasComplex*   x,
                                int               incx,
                                hipblasComplex    beta,
                                hipblasComplex*   y,
                                int               incy)
{
    cblas_chbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_hbmv<hipblasDoubleComplex>(hipblasFillMode_t     uplo,
                                      int                   n,
                                      int                   k,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      hipblasDoubleComplex* x,
                                      int                   incx,
                                      hipblasDoubleComplex  beta,
                                      hipblasDoubleComplex* y,
                                      int                   incy)
{
    cblas_zhbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

// hemv
template <>
void cblas_hemv<hipblasComplex>(hipblasFillMode_t uplo,
                                int               n,
                                hipblasComplex    alpha,
                                hipblasComplex*   A,
                                int               lda,
                                hipblasComplex*   x,
                                int               incx,
                                hipblasComplex    beta,
                                hipblasComplex*   y,
                                int               incy)
{
    cblas_chemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
void cblas_hemv<hipblasDoubleComplex>(hipblasFillMode_t     uplo,
                                      int                   n,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      hipblasDoubleComplex* x,
                                      int                   incx,
                                      hipblasDoubleComplex  beta,
                                      hipblasDoubleComplex* y,
                                      int                   incy)
{
    cblas_zhemv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// her
template <>
void cblas_her<hipblasComplex, float>(hipblasFillMode_t uplo,
                                      int               n,
                                      float             alpha,
                                      hipblasComplex*   x,
                                      int               incx,
                                      hipblasComplex*   A,
                                      int               lda)
{
    cblas_cher(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

template <>
void cblas_her<hipblasDoubleComplex, double>(hipblasFillMode_t     uplo,
                                             int                   n,
                                             double                alpha,
                                             hipblasDoubleComplex* x,
                                             int                   incx,
                                             hipblasDoubleComplex* A,
                                             int                   lda)
{
    cblas_zher(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, A, lda);
}

// her2
template <>
void cblas_her2<hipblasComplex>(hipblasFillMode_t uplo,
                                int               n,
                                hipblasComplex    alpha,
                                hipblasComplex*   x,
                                int               incx,
                                hipblasComplex*   y,
                                int               incy,
                                hipblasComplex*   A,
                                int               lda)
{
    cblas_cher2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_her2<hipblasDoubleComplex>(hipblasFillMode_t     uplo,
                                      int                   n,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* x,
                                      int                   incx,
                                      hipblasDoubleComplex* y,
                                      int                   incy,
                                      hipblasDoubleComplex* A,
                                      int                   lda)
{
    cblas_zher2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, A, lda);
}

// hpmv
template <>
void cblas_hpmv<hipblasComplex>(hipblasFillMode_t uplo,
                                int               n,
                                hipblasComplex    alpha,
                                hipblasComplex*   AP,
                                hipblasComplex*   x,
                                int               incx,
                                hipblasComplex    beta,
                                hipblasComplex*   y,
                                int               incy)
{
    cblas_chpmv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, AP, x, incx, &beta, y, incy);
}

template <>
void cblas_hpmv<hipblasDoubleComplex>(hipblasFillMode_t     uplo,
                                      int                   n,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* AP,
                                      hipblasDoubleComplex* x,
                                      int                   incx,
                                      hipblasDoubleComplex  beta,
                                      hipblasDoubleComplex* y,
                                      int                   incy)
{
    cblas_zhpmv(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, AP, x, incx, &beta, y, incy);
}

// hpr
template <>
void cblas_hpr(
    hipblasFillMode_t uplo, int n, float alpha, hipblasComplex* x, int incx, hipblasComplex* AP)
{
    cblas_chpr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

template <>
void cblas_hpr(hipblasFillMode_t     uplo,
               int                   n,
               double                alpha,
               hipblasDoubleComplex* x,
               int                   incx,
               hipblasDoubleComplex* AP)
{
    cblas_zhpr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

// hpr2
template <>
void cblas_hpr2(hipblasFillMode_t uplo,
                int               n,
                hipblasComplex    alpha,
                hipblasComplex*   x,
                int               incx,
                hipblasComplex*   y,
                int               incy,
                hipblasComplex*   AP)
{
    cblas_chpr2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, AP);
}

template <>
void cblas_hpr2(hipblasFillMode_t     uplo,
                int                   n,
                hipblasDoubleComplex  alpha,
                hipblasDoubleComplex* x,
                int                   incx,
                hipblasDoubleComplex* y,
                int                   incy,
                hipblasDoubleComplex* AP)
{
    cblas_zhpr2(CblasColMajor, (CBLAS_UPLO)uplo, n, &alpha, x, incx, y, incy, AP);
}

// sbmv
template <>
void cblas_sbmv(hipblasFillMode_t uplo,
                int               n,
                int               k,
                float             alpha,
                float*            A,
                int               lda,
                float*            x,
                int               incx,
                float             beta,
                float*            y,
                int               incy)
{
    cblas_ssbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
void cblas_sbmv(hipblasFillMode_t uplo,
                int               n,
                int               k,
                double            alpha,
                double*           A,
                int               lda,
                double*           x,
                int               incx,
                double            beta,
                double*           y,
                int               incy)
{
    cblas_dsbmv(CblasColMajor, (CBLAS_UPLO)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// spmv
template <>
void cblas_spmv(hipblasFillMode_t uplo,
                int               n,
                float             alpha,
                float*            AP,
                float*            x,
                int               incx,
                float             beta,
                float*            y,
                int               incy)
{
    cblas_sspmv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, AP, x, incx, beta, y, incy);
}

template <>
void cblas_spmv(hipblasFillMode_t uplo,
                int               n,
                double            alpha,
                double*           AP,
                double*           x,
                int               incx,
                double            beta,
                double*           y,
                int               incy)
{
    cblas_dspmv(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, AP, x, incx, beta, y, incy);
}

// spr
template <>
void cblas_spr(hipblasFillMode_t uplo, int n, float alpha, float* x, int incx, float* AP)
{
    cblas_sspr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

template <>
void cblas_spr(hipblasFillMode_t uplo, int n, double alpha, double* x, int incx, double* AP)
{
    cblas_dspr(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, AP);
}

template <>
void cblas_spr(hipblasFillMode_t uplo,
               int               n,
               hipblasComplex    alpha,
               hipblasComplex*   x,
               int               incx,
               hipblasComplex*   AP)
{
    char u = uplo == HIPBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    cspr_(&u, &n, &alpha, x, &incx, AP);
}

template <>
void cblas_spr(hipblasFillMode_t     uplo,
               int                   n,
               hipblasDoubleComplex  alpha,
               hipblasDoubleComplex* x,
               int                   incx,
               hipblasDoubleComplex* AP)
{
    char u = uplo == HIPBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    zspr_(&u, &n, &alpha, x, &incx, AP);
}

// spr2
template <>
void cblas_spr2(
    hipblasFillMode_t uplo, int n, float alpha, float* x, int incx, float* y, int incy, float* AP)
{
    cblas_sspr2(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, y, incy, AP);
}

template <>
void cblas_spr2(hipblasFillMode_t uplo,
                int               n,
                double            alpha,
                double*           x,
                int               incx,
                double*           y,
                int               incy,
                double*           AP)
{
    cblas_dspr2(CblasColMajor, (CBLAS_UPLO)uplo, n, alpha, x, incx, y, incy, AP);
}

// symv
template <>
void cblas_symv(hipblasFillMode_t uplo,
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
void cblas_symv(hipblasFillMode_t uplo,
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

template <>
void cblas_symv(hipblasFillMode_t uplo,
                int               n,
                hipblasComplex    alpha,
                hipblasComplex*   A,
                int               lda,
                hipblasComplex*   x,
                int               incx,
                hipblasComplex    beta,
                hipblasComplex*   y,
                int               incy)
{
    char u = uplo == HIPBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    csymv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cblas_symv(hipblasFillMode_t     uplo,
                int                   n,
                hipblasDoubleComplex  alpha,
                hipblasDoubleComplex* A,
                int                   lda,
                hipblasDoubleComplex* x,
                int                   incx,
                hipblasDoubleComplex  beta,
                hipblasDoubleComplex* y,
                int                   incy)
{
    char u = uplo == HIPBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    zsymv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
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

template <>
void cblas_syr<hipblasComplex>(hipblasFillMode_t uplo,
                               int               n,
                               hipblasComplex    alpha,
                               hipblasComplex*   x,
                               int               incx,
                               hipblasComplex*   A,
                               int               lda)
{
    char u = uplo == HIPBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    csyr_(&u, &n, &alpha, x, &incx, A, &lda);
}

template <>
void cblas_syr<hipblasDoubleComplex>(hipblasFillMode_t     uplo,
                                     int                   n,
                                     hipblasDoubleComplex  alpha,
                                     hipblasDoubleComplex* x,
                                     int                   incx,
                                     hipblasDoubleComplex* A,
                                     int                   lda)
{
    char u = uplo == HIPBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    zsyr_(&u, &n, &alpha, x, &incx, A, &lda);
}

// syr2
// No complex version of syr2 - make a local implementation
template <typename T>
void cblas_syr2_local(
    hipblasFillMode_t uplo, int n, T alpha, T* xa, int incx, T* ya, int incy, T* A, int lda)
{
    if(n <= 0)
        return;

    T* x = incx < 0 ? xa - ptrdiff_t(incx) * (n - 1) : xa;
    T* y = incy < 0 ? ya - ptrdiff_t(incy) * (n - 1) : ya;

    if(uplo == HIPBLAS_FILL_MODE_UPPER)
        for(int j = 0; j < n; ++j)
        {
            T tmpx = alpha * x[j * incx];
            T tmpy = alpha * y[j * incx];
            for(int i = 0; i <= j; ++i)
                A[i + j * lda] += x[i * incx] * tmpy + y[i * incy] * tmpx;
        }
    else
        for(int j = 0; j < n; ++j)
        {
            T tmpx = alpha * x[j * incx];
            T tmpy = alpha * y[j * incx];
            for(int i = j; i < n; ++i)
                A[i + j * lda] += x[i * incx] * tmpy + y[i * incy] * tmpx;
        }
}

template <>
void cblas_syr2(hipblasFillMode_t uplo,
                int               n,
                float             alpha,
                float*            x,
                int               incx,
                float*            y,
                int               incy,
                float*            A,
                int               lda)
{
    cblas_ssyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_syr2(hipblasFillMode_t uplo,
                int               n,
                double            alpha,
                double*           x,
                int               incx,
                double*           y,
                int               incy,
                double*           A,
                int               lda)
{
    cblas_dsyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_syr2(hipblasFillMode_t uplo,
                int               n,
                hipblasComplex    alpha,
                hipblasComplex*   x,
                int               incx,
                hipblasComplex*   y,
                int               incy,
                hipblasComplex*   A,
                int               lda)
{
    cblas_syr2_local(uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
void cblas_syr2(hipblasFillMode_t     uplo,
                int                   n,
                hipblasDoubleComplex  alpha,
                hipblasDoubleComplex* x,
                int                   incx,
                hipblasDoubleComplex* y,
                int                   incy,
                hipblasDoubleComplex* A,
                int                   lda)
{
    cblas_syr2_local(uplo, n, alpha, x, incx, y, incy, A, lda);
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

template <>
int cblas_potrf(char uplo, int m, hipblasComplex* A, int lda)
{
    int info;
    cpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

template <>
int cblas_potrf(char uplo, int m, hipblasDoubleComplex* A, int lda)
{
    int info;
    zpotrf_(&uplo, &m, A, &lda, &info);
    return info;
}

// tbmv
template <>
void cblas_tbmv<float>(hipblasFillMode_t  uplo,
                       hipblasOperation_t transA,
                       hipblasDiagType_t  diag,
                       int                m,
                       int                k,
                       const float*       A,
                       int                lda,
                       float*             x,
                       int                incx)
{
    cblas_stbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_tbmv<double>(hipblasFillMode_t  uplo,
                        hipblasOperation_t transA,
                        hipblasDiagType_t  diag,
                        int                m,
                        int                k,
                        const double*      A,
                        int                lda,
                        double*            x,
                        int                incx)
{
    cblas_dtbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_tbmv<hipblasComplex>(hipblasFillMode_t     uplo,
                                hipblasOperation_t    transA,
                                hipblasDiagType_t     diag,
                                int                   m,
                                int                   k,
                                const hipblasComplex* A,
                                int                   lda,
                                hipblasComplex*       x,
                                int                   incx)
{
    cblas_ctbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_tbmv<hipblasDoubleComplex>(hipblasFillMode_t           uplo,
                                      hipblasOperation_t          transA,
                                      hipblasDiagType_t           diag,
                                      int                         m,
                                      int                         k,
                                      const hipblasDoubleComplex* A,
                                      int                         lda,
                                      hipblasDoubleComplex*       x,
                                      int                         incx)
{
    cblas_ztbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

// tbsv
template <>
void cblas_tbsv<float>(hipblasFillMode_t  uplo,
                       hipblasOperation_t transA,
                       hipblasDiagType_t  diag,
                       int                m,
                       int                k,
                       const float*       A,
                       int                lda,
                       float*             x,
                       int                incx)
{
    cblas_stbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_tbsv<double>(hipblasFillMode_t  uplo,
                        hipblasOperation_t transA,
                        hipblasDiagType_t  diag,
                        int                m,
                        int                k,
                        const double*      A,
                        int                lda,
                        double*            x,
                        int                incx)
{
    cblas_dtbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_tbsv<hipblasComplex>(hipblasFillMode_t     uplo,
                                hipblasOperation_t    transA,
                                hipblasDiagType_t     diag,
                                int                   m,
                                int                   k,
                                const hipblasComplex* A,
                                int                   lda,
                                hipblasComplex*       x,
                                int                   incx)
{
    cblas_ctbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
void cblas_tbsv<hipblasDoubleComplex>(hipblasFillMode_t           uplo,
                                      hipblasOperation_t          transA,
                                      hipblasDiagType_t           diag,
                                      int                         m,
                                      int                         k,
                                      const hipblasDoubleComplex* A,
                                      int                         lda,
                                      hipblasDoubleComplex*       x,
                                      int                         incx)
{
    cblas_ztbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

// tpmv
template <>
void cblas_tpmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const float*       A,
                float*             x,
                int                incx)
{
    cblas_stpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
void cblas_tpmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const double*      A,
                double*            x,
                int                incx)
{
    cblas_dtpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
void cblas_tpmv(hipblasFillMode_t     uplo,
                hipblasOperation_t    transA,
                hipblasDiagType_t     diag,
                int                   m,
                const hipblasComplex* A,
                hipblasComplex*       x,
                int                   incx)
{
    cblas_ctpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
void cblas_tpmv(hipblasFillMode_t           uplo,
                hipblasOperation_t          transA,
                hipblasDiagType_t           diag,
                int                         m,
                const hipblasDoubleComplex* A,
                hipblasDoubleComplex*       x,
                int                         incx)
{
    cblas_ztpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

// tpsv
template <>
void cblas_tpsv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                n,
                const float*       AP,
                float*             x,
                int                incx)
{
    cblas_stpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
void cblas_tpsv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                n,
                const double*      AP,
                double*            x,
                int                incx)
{
    cblas_dtpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
void cblas_tpsv(hipblasFillMode_t     uplo,
                hipblasOperation_t    transA,
                hipblasDiagType_t     diag,
                int                   n,
                const hipblasComplex* AP,
                hipblasComplex*       x,
                int                   incx)
{
    cblas_ctpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
void cblas_tpsv(hipblasFillMode_t           uplo,
                hipblasOperation_t          transA,
                hipblasDiagType_t           diag,
                int                         n,
                const hipblasDoubleComplex* AP,
                hipblasDoubleComplex*       x,
                int                         incx)
{
    cblas_ztpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

// trmv
template <>
void cblas_trmv<float>(hipblasFillMode_t  uplo,
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
void cblas_trmv<double>(hipblasFillMode_t  uplo,
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

template <>
void cblas_trmv<hipblasComplex>(hipblasFillMode_t     uplo,
                                hipblasOperation_t    transA,
                                hipblasDiagType_t     diag,
                                int                   m,
                                const hipblasComplex* A,
                                int                   lda,
                                hipblasComplex*       x,
                                int                   incx)
{
    cblas_ctrmv(CblasColMajor,
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
void cblas_trmv<hipblasDoubleComplex>(hipblasFillMode_t           uplo,
                                      hipblasOperation_t          transA,
                                      hipblasDiagType_t           diag,
                                      int                         m,
                                      const hipblasDoubleComplex* A,
                                      int                         lda,
                                      hipblasDoubleComplex*       x,
                                      int                         incx)
{
    cblas_ztrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
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

template <>
void cblas_trsv<hipblasComplex>(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                hipblasOperation_t    transA,
                                hipblasDiagType_t     diag,
                                int                   m,
                                const hipblasComplex* A,
                                int                   lda,
                                hipblasComplex*       x,
                                int                   incx)
{
    cblas_ctrsv(CblasColMajor,
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
void cblas_trsv<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                      hipblasFillMode_t           uplo,
                                      hipblasOperation_t          transA,
                                      hipblasDiagType_t           diag,
                                      int                         m,
                                      const hipblasDoubleComplex* A,
                                      int                         lda,
                                      hipblasDoubleComplex*       x,
                                      int                         incx)
{
    cblas_ztrsv(CblasColMajor,
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

template <typename T>
void cblas_geam_helper(hipblasOperation_t transA,
                       hipblasOperation_t transB,
                       int                M,
                       int                N,
                       T                  alpha,
                       T*                 A,
                       int                lda,
                       T                  beta,
                       T*                 B,
                       int                ldb,
                       T*                 C,
                       int                ldc)
{
    int inc1_A = transA == HIPBLAS_OP_N ? 1 : lda;
    int inc2_A = transA == HIPBLAS_OP_N ? lda : 1;
    int inc1_B = transB == HIPBLAS_OP_N ? 1 : ldb;
    int inc2_B = transB == HIPBLAS_OP_N ? ldb : 1;

    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            T a_val = A[i * inc1_A + j * inc2_A];
            T b_val = B[i * inc1_B + j * inc2_B];
            if(transA == HIPBLAS_OP_C)
                a_val = std::conj(a_val);
            if(transB == HIPBLAS_OP_C)
                b_val = std::conj(b_val);
            C[i + j * ldc] = alpha * a_val + beta * b_val;
        }
    }
}

// geam
template <>
void cblas_geam(hipblasOperation_t transa,
                hipblasOperation_t transb,
                int                m,
                int                n,
                float*             alpha,
                float*             A,
                int                lda,
                float*             beta,
                float*             B,
                int                ldb,
                float*             C,
                int                ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void cblas_geam(hipblasOperation_t transa,
                hipblasOperation_t transb,
                int                m,
                int                n,
                double*            alpha,
                double*            A,
                int                lda,
                double*            beta,
                double*            B,
                int                ldb,
                double*            C,
                int                ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void cblas_geam(hipblasOperation_t transa,
                hipblasOperation_t transb,
                int                m,
                int                n,
                hipblasComplex*    alpha,
                hipblasComplex*    A,
                int                lda,
                hipblasComplex*    beta,
                hipblasComplex*    B,
                int                ldb,
                hipblasComplex*    C,
                int                ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void cblas_geam(hipblasOperation_t    transa,
                hipblasOperation_t    transb,
                int                   m,
                int                   n,
                hipblasDoubleComplex* alpha,
                hipblasDoubleComplex* A,
                int                   lda,
                hipblasDoubleComplex* beta,
                hipblasDoubleComplex* B,
                int                   ldb,
                hipblasDoubleComplex* C,
                int                   ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

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
void cblas_gemm<hipblasHalf, hipblasHalf, float>(hipblasOperation_t transA,
                                                 hipblasOperation_t transB,
                                                 int                m,
                                                 int                n,
                                                 int                k,
                                                 float              alpha_float,
                                                 hipblasHalf*       A,
                                                 int                lda,
                                                 hipblasHalf*       B,
                                                 int                ldb,
                                                 float              beta_float,
                                                 hipblasHalf*       C,
                                                 int                ldc)
{
    // cblas does not support hipblasHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

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
void cblas_gemm<hipblasComplex>(hipblasOperation_t transA,
                                hipblasOperation_t transB,
                                int                m,
                                int                n,
                                int                k,
                                hipblasComplex     alpha,
                                hipblasComplex*    A,
                                int                lda,
                                hipblasComplex*    B,
                                int                ldb,
                                hipblasComplex     beta,
                                hipblasComplex*    C,
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
void cblas_gemm<hipblasDoubleComplex>(hipblasOperation_t    transA,
                                      hipblasOperation_t    transB,
                                      int                   m,
                                      int                   n,
                                      int                   k,
                                      hipblasDoubleComplex  alpha,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      hipblasDoubleComplex* B,
                                      int                   ldb,
                                      hipblasDoubleComplex  beta,
                                      hipblasDoubleComplex* C,
                                      int                   ldc)
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

template <>
void cblas_gemm<int8_t, int32_t, int32_t>(hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          int                k,
                                          int32_t            alpha,
                                          int8_t*            A,
                                          int                lda,
                                          int8_t*            B,
                                          int                ldb,
                                          int32_t            beta,
                                          int32_t*           C,
                                          int                ldc)
{
    double alpha_double = static_cast<double>(alpha);
    double beta_double  = static_cast<double>(beta);

    size_t const sizeA = ((transA == HIPBLAS_OP_N) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == HIPBLAS_OP_N) ? n : k) * size_t(ldb);
    size_t const sizeC = n * size_t(ldc);

    std::unique_ptr<double[]> A_double(new double[sizeA]());
    std::unique_ptr<double[]> B_double(new double[sizeB]());
    std::unique_ptr<double[]> C_double(new double[sizeC]());

    for(int i = 0; i < sizeA; i++)
    {
        A_double[i] = static_cast<double>(A[i]);
    }
    for(int i = 0; i < sizeB; i++)
    {
        B_double[i] = static_cast<double>(B[i]);
    }
    for(int i = 0; i < sizeC; i++)
    {
        C_double[i] = static_cast<double>(C[i]);
    }

    cblas_dgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha_double,
                const_cast<const double*>(A_double.get()),
                lda,
                const_cast<const double*>(B_double.get()),
                ldb,
                beta_double,
                static_cast<double*>(C_double.get()),
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<int32_t>(C_double[i]);
}

// hemm
template <>
void cblas_hemm(hipblasSideMode_t side,
                hipblasFillMode_t uplo,
                int               m,
                int               n,
                hipblasComplex    alpha,
                hipblasComplex*   A,
                int               lda,
                hipblasComplex*   B,
                int               ldb,
                hipblasComplex    beta,
                hipblasComplex*   C,
                int               ldc)
{
    cblas_chemm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
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
void cblas_hemm(hipblasSideMode_t     side,
                hipblasFillMode_t     uplo,
                int                   m,
                int                   n,
                hipblasDoubleComplex  alpha,
                hipblasDoubleComplex* A,
                int                   lda,
                hipblasDoubleComplex* B,
                int                   ldb,
                hipblasDoubleComplex  beta,
                hipblasDoubleComplex* C,
                int                   ldc)
{
    cblas_zhemm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// herk
template <>
void cblas_herk(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                int                n,
                int                k,
                float              alpha,
                hipblasComplex*    A,
                int                lda,
                float              beta,
                hipblasComplex*    C,
                int                ldc)
{
    cblas_cherk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
void cblas_herk(hipblasFillMode_t     uplo,
                hipblasOperation_t    transA,
                int                   n,
                int                   k,
                double                alpha,
                hipblasDoubleComplex* A,
                int                   lda,
                double                beta,
                hipblasDoubleComplex* C,
                int                   ldc)
{
    cblas_zherk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

// herkx
template <typename T, typename U>
void cblas_herkx_local(hipblasFillMode_t  uplo,
                       hipblasOperation_t transA,
                       int                n,
                       int                k,
                       T                  alpha,
                       T*                 A,
                       int                lda,
                       T*                 B,
                       int                ldb,
                       U                  beta,
                       T*                 C,
                       int                ldc)
{

    if(n <= 0 || (beta == 1 && (k == 0 || alpha == T(0))))
        return;

    if(transA == HIPBLAS_OP_N)
    {
        if(uplo == HIPBLAS_FILL_MODE_UPPER)
        {
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i <= j; i++)
                    C[i + j * ldc] *= T(beta);

                for(int l = 0; l < k; l++)
                {
                    T temp = alpha * std::conj(B[j + l * ldb]);
                    for(int i = 0; i <= j; ++i)
                        C[i + j * ldc] += temp * A[i + l * lda];
                }
            }
        }
        else // lower
        {
            for(int j = 0; j < n; ++j)
            {
                for(int i = j; i < n; i++)
                    C[i + j * ldc] *= T(beta);

                for(int l = 0; l < k; l++)
                {
                    T temp = alpha * std::conj(B[j + l * ldb]);
                    for(int i = j; i < n; ++i)
                        C[i + j * ldc] += temp * A[i + l * lda];
                }
            }
        }
    }
    else // conjugate transpose
    {
        if(uplo == HIPBLAS_FILL_MODE_UPPER)
        {
            for(int j = 0; j < n; ++j)
                for(int i = 0; i <= j; i++)
                {
                    C[i + j * ldc] *= T(beta);
                    T temp(0);
                    for(int l = 0; l < k; l++)
                        temp += std::conj(A[l + i * lda]) * B[l + j * ldb];
                    C[i + j * ldc] += alpha * temp;
                }
        }
        else // lower
        {
            for(int j = 0; j < n; ++j)
                for(int i = j; i < n; i++)
                {
                    C[i + j * ldc] *= T(beta);
                    T temp(0);
                    for(int l = 0; l < k; l++)
                        temp += std::conj(A[l + i * lda]) * B[l + j * ldb];
                    C[i + j * ldc] += alpha * temp;
                }
        }
    }

    for(int i = 0; i < n; i++)
        C[i + i * ldc].imag(0);
}

template <>
void cblas_herkx(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
                 int                n,
                 int                k,
                 hipblasComplex     alpha,
                 hipblasComplex*    A,
                 int                lda,
                 hipblasComplex*    B,
                 int                ldb,
                 float              beta,
                 hipblasComplex*    C,
                 int                ldc)
{
    cblas_herkx_local(uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void cblas_herkx(hipblasFillMode_t     uplo,
                 hipblasOperation_t    transA,
                 int                   n,
                 int                   k,
                 hipblasDoubleComplex  alpha,
                 hipblasDoubleComplex* A,
                 int                   lda,
                 hipblasDoubleComplex* B,
                 int                   ldb,
                 double                beta,
                 hipblasDoubleComplex* C,
                 int                   ldc)
{
    cblas_herkx_local(uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// her2k
template <>
void cblas_her2k(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
                 int                n,
                 int                k,
                 hipblasComplex     alpha,
                 hipblasComplex*    A,
                 int                lda,
                 hipblasComplex*    B,
                 int                ldb,
                 float              beta,
                 hipblasComplex*    C,
                 int                ldc)
{
    cblas_cher2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 &alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
void cblas_her2k(hipblasFillMode_t     uplo,
                 hipblasOperation_t    transA,
                 int                   n,
                 int                   k,
                 hipblasDoubleComplex  alpha,
                 hipblasDoubleComplex* A,
                 int                   lda,
                 hipblasDoubleComplex* B,
                 int                   ldb,
                 double                beta,
                 hipblasDoubleComplex* C,
                 int                   ldc)
{
    cblas_zher2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
                 n,
                 k,
                 &alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

// symm
template <>
void cblas_symm(hipblasSideMode_t side,
                hipblasFillMode_t uplo,
                int               m,
                int               n,
                float             alpha,
                float*            A,
                int               lda,
                float*            B,
                int               ldb,
                float             beta,
                float*            C,
                int               ldc)
{
    cblas_ssymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
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
void cblas_symm(hipblasSideMode_t side,
                hipblasFillMode_t uplo,
                int               m,
                int               n,
                double            alpha,
                double*           A,
                int               lda,
                double*           B,
                int               ldb,
                double            beta,
                double*           C,
                int               ldc)
{
    cblas_dsymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
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
void cblas_symm(hipblasSideMode_t side,
                hipblasFillMode_t uplo,
                int               m,
                int               n,
                hipblasComplex    alpha,
                hipblasComplex*   A,
                int               lda,
                hipblasComplex*   B,
                int               ldb,
                hipblasComplex    beta,
                hipblasComplex*   C,
                int               ldc)
{
    cblas_csymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
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
void cblas_symm(hipblasSideMode_t     side,
                hipblasFillMode_t     uplo,
                int                   m,
                int                   n,
                hipblasDoubleComplex  alpha,
                hipblasDoubleComplex* A,
                int                   lda,
                hipblasDoubleComplex* B,
                int                   ldb,
                hipblasDoubleComplex  beta,
                hipblasDoubleComplex* C,
                int                   ldc)
{
    cblas_zsymm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// syrk
template <>
void cblas_syrk(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                int                n,
                int                k,
                float              alpha,
                float*             A,
                int                lda,
                float              beta,
                float*             C,
                int                ldc)
{
    cblas_ssyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
void cblas_syrk(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                int                n,
                int                k,
                double             alpha,
                double*            A,
                int                lda,
                double             beta,
                double*            C,
                int                ldc)
{
    cblas_dsyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
}

template <>
void cblas_syrk(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                int                n,
                int                k,
                hipblasComplex     alpha,
                hipblasComplex*    A,
                int                lda,
                hipblasComplex     beta,
                hipblasComplex*    C,
                int                ldc)
{
    cblas_csyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                &alpha,
                A,
                lda,
                &beta,
                C,
                ldc);
}

template <>
void cblas_syrk(hipblasFillMode_t     uplo,
                hipblasOperation_t    transA,
                int                   n,
                int                   k,
                hipblasDoubleComplex  alpha,
                hipblasDoubleComplex* A,
                int                   lda,
                hipblasDoubleComplex  beta,
                hipblasDoubleComplex* C,
                int                   ldc)
{
    cblas_zsyrk(CblasColMajor,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                n,
                k,
                &alpha,
                A,
                lda,
                &beta,
                C,
                ldc);
}

// syr2k
template <>
void cblas_syr2k(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
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
    cblas_ssyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
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
void cblas_syr2k(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
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
    cblas_dsyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
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
void cblas_syr2k(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
                 int                n,
                 int                k,
                 hipblasComplex     alpha,
                 hipblasComplex*    A,
                 int                lda,
                 hipblasComplex*    B,
                 int                ldb,
                 hipblasComplex     beta,
                 hipblasComplex*    C,
                 int                ldc)
{
    cblas_csyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
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
void cblas_syr2k(hipblasFillMode_t     uplo,
                 hipblasOperation_t    transA,
                 int                   n,
                 int                   k,
                 hipblasDoubleComplex  alpha,
                 hipblasDoubleComplex* A,
                 int                   lda,
                 hipblasDoubleComplex* B,
                 int                   ldb,
                 hipblasDoubleComplex  beta,
                 hipblasDoubleComplex* C,
                 int                   ldc)
{
    cblas_zsyr2k(CblasColMajor,
                 (CBLAS_UPLO)uplo,
                 (CBLAS_TRANSPOSE)transA,
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

// syrkx
// Use syrk with A == B for now.

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

template <>
void cblas_trsm<hipblasComplex>(hipblasSideMode_t     side,
                                hipblasFillMode_t     uplo,
                                hipblasOperation_t    transA,
                                hipblasDiagType_t     diag,
                                int                   m,
                                int                   n,
                                hipblasComplex        alpha,
                                const hipblasComplex* A,
                                int                   lda,
                                hipblasComplex*       B,
                                int                   ldb)
{
    cblas_ctrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trsm<hipblasDoubleComplex>(hipblasSideMode_t           side,
                                      hipblasFillMode_t           uplo,
                                      hipblasOperation_t          transA,
                                      hipblasDiagType_t           diag,
                                      int                         m,
                                      int                         n,
                                      hipblasDoubleComplex        alpha,
                                      const hipblasDoubleComplex* A,
                                      int                         lda,
                                      hipblasDoubleComplex*       B,
                                      int                         ldb)
{
    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

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

template <>
int cblas_trtri<hipblasComplex>(char uplo, char diag, int n, hipblasComplex* A, int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    int info;
    ctrtri_(&uplo, &diag, &n, A, &lda, &info);
    return info;
}

template <>
int cblas_trtri<hipblasDoubleComplex>(char uplo, char diag, int n, hipblasDoubleComplex* A, int lda)
{
    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblas =%d, cblas=%d\n", transA, (CBLAS_TRANSPOSE)transA );
    int info;
    ztrtri_(&uplo, &diag, &n, A, &lda, &info);
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

template <>
void cblas_trmm<hipblasComplex>(hipblasSideMode_t     side,
                                hipblasFillMode_t     uplo,
                                hipblasOperation_t    transA,
                                hipblasDiagType_t     diag,
                                int                   m,
                                int                   n,
                                hipblasComplex        alpha,
                                const hipblasComplex* A,
                                int                   lda,
                                hipblasComplex*       B,
                                int                   ldb)
{
    cblas_ctrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

template <>
void cblas_trmm<hipblasDoubleComplex>(hipblasSideMode_t           side,
                                      hipblasFillMode_t           uplo,
                                      hipblasOperation_t          transA,
                                      hipblasDiagType_t           diag,
                                      int                         m,
                                      int                         n,
                                      hipblasDoubleComplex        alpha,
                                      const hipblasDoubleComplex* A,
                                      int                         lda,
                                      hipblasDoubleComplex*       B,
                                      int                         ldb)
{
    cblas_ztrmm(CblasColMajor,
                (CBLAS_SIDE)side,
                (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_DIAG)diag,
                m,
                n,
                &alpha,
                A,
                lda,
                B,
                ldb);
}

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

template <>
int cblas_getrf<hipblasComplex>(int m, int n, hipblasComplex* A, int lda, int* ipiv)
{
    int info;
    cgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

template <>
int cblas_getrf<hipblasDoubleComplex>(int m, int n, hipblasDoubleComplex* A, int lda, int* ipiv)
{
    int info;
    zgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// getrs
template <>
int cblas_getrs<float>(char trans, int n, int nrhs, float* A, int lda, int* ipiv, float* B, int ldb)
{
    int info;
    sgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    return info;
}

template <>
int cblas_getrs<double>(
    char trans, int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb)
{
    int info;
    dgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    return info;
}

template <>
int cblas_getrs<hipblasComplex>(
    char trans, int n, int nrhs, hipblasComplex* A, int lda, int* ipiv, hipblasComplex* B, int ldb)
{
    int info;
    cgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    return info;
}

template <>
int cblas_getrs<hipblasDoubleComplex>(char                  trans,
                                      int                   n,
                                      int                   nrhs,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      int*                  ipiv,
                                      hipblasDoubleComplex* B,
                                      int                   ldb)
{
    int info;
    zgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    return info;
}

// getri
template <>
int cblas_getri<float>(int n, float* A, int lda, int* ipiv, float* work, int lwork)
{
    int info;
    sgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    return info;
}

template <>
int cblas_getri<double>(int n, double* A, int lda, int* ipiv, double* work, int lwork)
{
    int info;
    dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    return info;
}

template <>
int cblas_getri<hipblasComplex>(
    int n, hipblasComplex* A, int lda, int* ipiv, hipblasComplex* work, int lwork)
{
    int info;
    cgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    return info;
}

template <>
int cblas_getri<hipblasDoubleComplex>(
    int n, hipblasDoubleComplex* A, int lda, int* ipiv, hipblasDoubleComplex* work, int lwork)
{
    int info;
    zgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    return info;
}

// geqrf
template <>
int cblas_geqrf<float>(int m, int n, float* A, int lda, float* tau, float* work, int lwork)
{
    int info;
    sgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    return info;
}

template <>
int cblas_geqrf<double>(int m, int n, double* A, int lda, double* tau, double* work, int lwork)
{
    int info;
    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    return info;
}
template <>
int cblas_geqrf<hipblasComplex>(
    int m, int n, hipblasComplex* A, int lda, hipblasComplex* tau, hipblasComplex* work, int lwork)
{
    int info;
    cgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    return info;
}

template <>
int cblas_geqrf<hipblasDoubleComplex>(int                   m,
                                      int                   n,
                                      hipblasDoubleComplex* A,
                                      int                   lda,
                                      hipblasDoubleComplex* tau,
                                      hipblasDoubleComplex* work,
                                      int                   lwork)
{
    int info;
    zgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    return info;
}
