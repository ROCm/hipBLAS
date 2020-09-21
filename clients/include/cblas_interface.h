/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
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
void cblas_axpy(int n, const T alpha, const T* x, int incx, T* y, int incy);
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

template <typename T1, typename T2 = T1, typename T3 = T1>
void cblas_rot(int n, T1* x, int incx, T1* y, int incy, T2 c, T3 s);

template <typename T1, typename T2 = T1>
void cblas_rotg(T1* a, T1* b, T2* c, T1* s);

template <typename T1>
void cblas_rotm(int n, T1* x, int incx, T1* y, int incy, T1* param);

template <typename T1>
void cblas_rotmg(T1* d1, T1* d2, T1* x1, T1* y1, T1* param);

template <typename T1, typename T2>
void cblas_asum(int n, const T1* x, int incx, T2* result);

template <typename T>
void cblas_iamax(int n, const T* x, int incx, int* result);

template <typename T>
void cblas_iamin(int n, const T* x, int incx, int* result);

template <typename T>
void cblas_gbmv(hipblasOperation_t transA,
                int                m,
                int                n,
                int                kl,
                int                ku,
                T                  alpha,
                T*                 A,
                int                lda,
                T*                 x,
                int                incx,
                T                  beta,
                T*                 y,
                int                incy);

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

// ger (ger, geru, gerc)
template <typename T, bool CONJ>
void cblas_ger(int m, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

// hbmv
template <typename T>
void cblas_hbmv(hipblasFillMode_t uplo,
                int               n,
                int               k,
                T                 alpha,
                T*                A,
                int               lda,
                T*                x,
                int               incx,
                T                 beta,
                T*                y,
                int               incy);

// hemv
template <typename T, typename U>
void cblas_hemv(
    hipblasFillMode_t uplo, int n, U alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// spr
template <typename T>
void cblas_spr(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* AP);

// spr2
template <typename T>
void cblas_spr2(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* AP);

// syr
template <typename T>
void cblas_syr(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* A, int lda);

// syr2
template <typename T>
void cblas_syr2(
    hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

// her
template <typename T, typename U>
void cblas_her(hipblasFillMode_t uplo, int n, U alpha, T* x, int incx, T* A, int lda);

// her2
template <typename T>
void cblas_her2(
    hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

// hpmv
template <typename T>
void cblas_hpmv(
    hipblasFillMode_t uplo, int n, T alpha, T* AP, T* x, int incx, T beta, T* y, int incy);

// hpr
template <typename T, typename U>
void cblas_hpr(hipblasFillMode_t uplo, int n, U alpha, T* x, int incx, T* AP);

// hpr2
template <typename T>
void cblas_hpr2(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* AP);

// sbmv
template <typename T>
void cblas_sbmv(hipblasFillMode_t uplo,
                int               n,
                int               k,
                T                 alpha,
                T*                A,
                int               lda,
                T*                x,
                int               incx,
                T                 beta,
                T*                y,
                int               incy);

// spmv
template <typename T>
void cblas_spmv(
    hipblasFillMode_t uplo, int n, T alpha, T* AP, T* x, int incx, T beta, T* y, int incy);

// symv
template <typename T>
void cblas_symv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// potrf
template <typename T>
int cblas_potrf(char uplo, int m, T* A, int lda);

// tbmv
template <typename T>
void cblas_tbmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                int                k,
                const T*           A,
                int                lda,
                T*                 x,
                int                incx);

// tbsv
template <typename T>
void cblas_tbsv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                int                k,
                const T*           A,
                int                lda,
                T*                 x,
                int                incx);

// tpmv
template <typename T>
void cblas_tpmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const T*           A,
                T*                 x,
                int                incx);

// tpsv
template <typename T>
void cblas_tpsv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                n,
                const T*           AP,
                T*                 x,
                int                incx);

// trmv
template <typename T>
void cblas_trmv(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const T*           A,
                int                lda,
                T*                 x,
                int                incx);

// trsv
template <typename T>
void cblas_trsv(hipblasHandle_t    handle,
                hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                hipblasDiagType_t  diag,
                int                m,
                const T*           A,
                int                lda,
                T*                 x,
                int                incx);

// hemv
template <typename T>
void cblas_hemv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// herk
template <typename T, typename U>
void cblas_herk(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                int                n,
                int                k,
                U                  alpha,
                T*                 A,
                int                lda,
                U                  beta,
                T*                 C,
                int                ldc);

// herkx
template <typename T, typename U>
void cblas_herkx(hipblasFillMode_t  uplo,
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
                 int                ldc);

// her2k
template <typename T, typename U>
void cblas_her2k(hipblasFillMode_t  uplo,
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
                 int                ldc);

// geam
template <typename T>
void cblas_geam(hipblasOperation_t transa,
                hipblasOperation_t transb,
                int                m,
                int                n,
                T*                 alpha,
                T*                 A,
                int                lda,
                T*                 beta,
                T*                 B,
                int                ldb,
                T*                 C,
                int                ldc);

// gemm
template <typename Ti, typename To = Ti, typename Tc = To>
void cblas_gemm(hipblasOperation_t transA,
                hipblasOperation_t transB,
                int                m,
                int                n,
                int                k,
                Tc                 alpha,
                Ti*                A,
                int                lda,
                Ti*                B,
                int                ldb,
                Tc                 beta,
                To*                C,
                int                ldc);

// hemm
template <typename T>
void cblas_hemm(hipblasSideMode_t side,
                hipblasFillMode_t uplo,
                int               m,
                int               n,
                T                 alpha,
                T*                A,
                int               lda,
                T*                B,
                int               ldb,
                T                 beta,
                T*                C,
                int               ldc);

// symm
template <typename T>
void cblas_symm(hipblasSideMode_t side,
                hipblasFillMode_t uplo,
                int               m,
                int               n,
                T                 alpha,
                T*                A,
                int               lda,
                T*                B,
                int               ldb,
                T                 beta,
                T*                C,
                int               ldc);

// syrk
template <typename T>
void cblas_syrk(hipblasFillMode_t  uplo,
                hipblasOperation_t transA,
                int                n,
                int                k,
                T                  alpha,
                T*                 A,
                int                lda,
                T                  beta,
                T*                 C,
                int                ldc);

// syr2k
template <typename T>
void cblas_syr2k(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
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

// syrkx
template <typename T>
void cblas_syrkx(hipblasFillMode_t  uplo,
                 hipblasOperation_t transA,
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

// trsm
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

// trtri
template <typename T>
int cblas_trtri(char uplo, char diag, int n, T* A, int lda);

// trmm
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

template <typename T>
int cblas_getrs(char trans, int n, int nrhs, T* A, int lda, int* ipiv, T* B, int ldb);

template <typename T>
int cblas_getri(int n, T* A, int lda, int* ipiv, T* work, int lwork);

template <typename T>
int cblas_geqrf(int m, int n, T* A, int lda, T* tau, T* work, int lwork);
/* ============================================================================================ */

#endif /* _CBLAS_INTERFACE_ */
