/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

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

template <typename Ta, typename Tx = Ta>
void ref_axpy(int n, const Ta alpha, const Tx* x, int incx, Tx* y, int incy);
template <typename T, typename U = T>
void ref_scal(int n, const U alpha, T* x, int incx);
template <typename T>
void ref_copy(int n, T* x, int incx, T* y, int incy);
template <typename T>
void ref_swap(int n, T* x, int incx, T* y, int incy);

template <typename T>
void ref_dot(int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T>
void ref_dotc(int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T1, typename T2>
void ref_nrm2(int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2 = T1, typename T3 = T1>
void ref_rot(int n, T1* x, int incx, T1* y, int incy, T2 c, T3 s);

template <typename T1, typename T2 = T1>
void ref_rotg(T1* a, T1* b, T2* c, T1* s);

template <typename T1>
void ref_rotm(int n, T1* x, int incx, T1* y, int incy, T1* param);

template <typename T1>
void ref_rotmg(T1* d1, T1* d2, T1* x1, T1* y1, T1* param);

template <typename T1, typename T2>
void ref_asum(int n, const T1* x, int incx, T2* result);

template <typename T>
void ref_iamax(int n, const T* x, int incx, int* result);

template <typename T>
void ref_iamin(int n, const T* x, int incx, int* result);

template <typename T>
void ref_gbmv(hipblasOperation_t transA,
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
void ref_gemv(hipblasOperation_t transA,
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
void ref_symv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// ger (ger, geru, gerc)
template <typename T, bool CONJ>
void ref_ger(int m, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

// hbmv
template <typename T>
void ref_hbmv(hipblasFillMode_t uplo,
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
void ref_hemv(
    hipblasFillMode_t uplo, int n, U alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// spr
template <typename T>
void ref_spr(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* AP);

// spr2
template <typename T>
void ref_spr2(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* AP);

// syr
template <typename T>
void ref_syr(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* A, int lda);

// syr2
template <typename T>
void ref_syr2(
    hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

// her
template <typename T, typename U>
void ref_her(hipblasFillMode_t uplo, int n, U alpha, T* x, int incx, T* A, int lda);

// her2
template <typename T>
void ref_her2(
    hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* A, int lda);

// hpmv
template <typename T>
void ref_hpmv(
    hipblasFillMode_t uplo, int n, T alpha, T* AP, T* x, int incx, T beta, T* y, int incy);

// hpr
template <typename T, typename U>
void ref_hpr(hipblasFillMode_t uplo, int n, U alpha, T* x, int incx, T* AP);

// hpr2
template <typename T>
void ref_hpr2(hipblasFillMode_t uplo, int n, T alpha, T* x, int incx, T* y, int incy, T* AP);

// sbmv
template <typename T>
void ref_sbmv(hipblasFillMode_t uplo,
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
void ref_spmv(
    hipblasFillMode_t uplo, int n, T alpha, T* AP, T* x, int incx, T beta, T* y, int incy);

// symv
template <typename T>
void ref_symv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// potrf
template <typename T>
int ref_potrf(char uplo, int m, T* A, int lda);

// tbmv
template <typename T>
void ref_tbmv(hipblasFillMode_t  uplo,
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
void ref_tbsv(hipblasFillMode_t  uplo,
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
void ref_tpmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int                m,
              const T*           A,
              T*                 x,
              int                incx);

// tpsv
template <typename T>
void ref_tpsv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int                n,
              const T*           AP,
              T*                 x,
              int                incx);

// trmv
template <typename T>
void ref_trmv(hipblasFillMode_t  uplo,
              hipblasOperation_t transA,
              hipblasDiagType_t  diag,
              int                m,
              const T*           A,
              int                lda,
              T*                 x,
              int                incx);

// trsv
template <typename T>
void ref_trsv(hipblasHandle_t    handle,
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
void ref_hemv(
    hipblasFillMode_t uplo, int n, T alpha, T* A, int lda, T* x, int incx, T beta, T* y, int incy);

// herk
template <typename T, typename U>
void ref_herk(hipblasFillMode_t  uplo,
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
void ref_herkx(hipblasFillMode_t  uplo,
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
void ref_her2k(hipblasFillMode_t  uplo,
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
void ref_geam(hipblasOperation_t transa,
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
void ref_gemm(hipblasOperation_t transA,
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
void ref_hemm(hipblasSideMode_t side,
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
void ref_symm(hipblasSideMode_t side,
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
void ref_syrk(hipblasFillMode_t  uplo,
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
void ref_syr2k(hipblasFillMode_t  uplo,
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
void ref_syrkx(hipblasFillMode_t  uplo,
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
void ref_trsm(hipblasSideMode_t  side,
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
int ref_trtri(char uplo, char diag, int n, T* A, int lda);

// trmm
template <typename T>
void ref_trmm(hipblasSideMode_t  side,
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
int ref_getrf(int m, int n, T* A, int lda, int* ipiv);

template <typename T>
int ref_getrs(char trans, int n, int nrhs, T* A, int lda, int* ipiv, T* B, int ldb);

template <typename T>
int ref_getri(int n, T* A, int lda, int* ipiv, T* work, int lwork);

template <typename T>
int ref_geqrf(int m, int n, T* A, int lda, T* tau, T* work, int lwork);

template <typename T>
int ref_gels(char trans, int m, int n, int nrhs, T* A, int lda, T* B, int ldb, T* work, int lwork);
/* ============================================================================================ */

#endif /* _CBLAS_INTERFACE_ */
