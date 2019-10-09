/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once
#ifndef _CBLAS_INTERFACE_
#define _CBLAS_INTERFACE_

#include "hipblas.h"

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
*/

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

template <typename T>
void cblas_axpy(int n , const T alpha, const T* x, int incx, T* y, int incy);
template <typename T, typename U = T>
void cblas_scal(int n, const U alpha, T* x, int incx);
template <typename T>
void cblas_copy(int n, T* x, int incx, T* y, int incy);
template <typename T>
void cblas_swap(int n, T* x, int incx, T* y, int incy);

template <typename T>
void cblas_dot(int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T>
void cblas_dotc(int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T1, typename T2>
void cblas_nrm2(int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2>
void cblas_asum(int n, const T1* x, int incx, T2* result);

template <typename T>
void cblas_iamax(int n, const T* x, int incx, int* result);

template <typename T>
void cblas_iamin(int n, const T* x, int incx, int* result);

template <typename T>
void cblas_gemv(hipblasOperation_t transA,
                int                m,
                int                n,
                T                  alpha,
                T*                 A,
                int                lda,
                T*                 x,
                int                incx,
                T                  beta,
                T*                 y,
                int                incy);

template <typename T>
void cblas_symv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

template <typename T>
void cblas_ger(int m, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

template <typename T>
void cblas_hemv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

template <typename T>
void cblas_gemm(hipblasOperation_t transA,
                hipblasOperation_t transB,
                int                m,
                int                n,
                int                k,
                T                  alpha,
                T*                 A,
                int                lda,
                T*                 B,
                int                ldb,
                T                  beta,
                T*                 C,
                int                ldc);

template <typename T>
void cblas_trsm(hipblasSideMode_t  side,
                hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                int                n,
                T                  alpha,
                const T*           A,
                int                lda,
                T*                 B,
                int                ldb);

template <typename T>
int cblas_trtri(char uplo, char diag, int n, T* A, int lda);

template <typename T>
void cblas_trmm(hipblasSideMode_t  side,
                hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                int                n,
                T                  alpha,
                const T*           A,
                int                lda,
                T*                 B,
                int                ldb);

template <typename T>
int cblas_getrf(int m, int n, T* A, int lda, int* ipiv);
/* ============================================================================================ */

#endif /* _CBLAS_INTERFACE_ */
