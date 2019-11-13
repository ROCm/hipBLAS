/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _HIPBLAS_HPP_
#define _HIPBLAS_HPP_

/* library headers */
#include "hipblas.h"

/*!\file
 * \brief hipblasTemplate_api.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based
 * NVIDIA GPUs.
 *  This file exposes C++ templated BLAS interface with only the precision templated.
*/

/*
 * ===========================================================================
 *   READEME: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */
template <typename T, typename U = T>
hipblasStatus_t hipblasScal(hipblasHandle_t handle, int n, const U* alpha, T* x, int incx);

template <typename T, typename U = T>
hipblasStatus_t hipblasScalBatched(
    hipblasHandle_t handle, int n, const U* alpha, T* const x[], int incx, int batch_count);

template <typename T, typename U = T>
hipblasStatus_t hipblasScalStridedBatched(
    hipblasHandle_t handle, int n, const U* alpha, T* x, int incx, int stridex, int batch_count);

template <typename T>
hipblasStatus_t hipblasCopy(hipblasHandle_t handle, int n, const T* x, int incx, T* y, int incy);

template <typename T>
hipblasStatus_t hipblasCopyBatched(hipblasHandle_t handle,
                                   int             n,
                                   const T* const  x[],
                                   int             incx,
                                   T* const        y[],
                                   int             incy,
                                   int             batch_count);

template <typename T>
hipblasStatus_t hipblasCopyStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const T*        x,
                                          int             incx,
                                          int             stridex,
                                          T*              y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count);

template <typename T>
hipblasStatus_t hipblasSwap(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy);

template <typename T>
hipblasStatus_t hipblasSwapBatched(
    hipblasHandle_t handle, int n, T* x[], int incx, T* y[], int incy, int batch_count);

template <typename T>
hipblasStatus_t hipblasSwapStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          T*              x,
                                          int             incx,
                                          int             stridex,
                                          T*              y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count);

template <typename T>
hipblasStatus_t hipblasDot(
    hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T>
hipblasStatus_t hipblasDotc(
    hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T>
hipblasStatus_t hipblasDotBatched(hipblasHandle_t handle,
                                  int             n,
                                  const T* const  x[],
                                  int             incx,
                                  const T* const  y[],
                                  int             incy,
                                  int             batch_count,
                                  T*              result);

template <typename T>
hipblasStatus_t hipblasDotcBatched(hipblasHandle_t handle,
                                   int             n,
                                   const T* const  x[],
                                   int             incx,
                                   const T* const  y[],
                                   int             incy,
                                   int             batch_count,
                                   T*              result);

template <typename T>
hipblasStatus_t hipblasDotStridedBatched(hipblasHandle_t handle,
                                         int             n,
                                         const T*        x,
                                         int             incx,
                                         int             stridex,
                                         const T*        y,
                                         int             incy,
                                         int             stridey,
                                         int             batch_count,
                                         T*              result);

template <typename T>
hipblasStatus_t hipblasDotcStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const T*        x,
                                          int             incx,
                                          int             stridex,
                                          const T*        y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count,
                                          T*              result);

template <typename T1, typename T2>
hipblasStatus_t hipblasAsum(hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasAsumBatched(
    hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasAsumStridedBatched(
    hipblasHandle_t handle, int n, const T1* x, int incx, int stridex, int batch_count, T2* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasNrm2(hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasNrm2Batched(
    hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasNrm2StridedBatched(
    hipblasHandle_t handle, int n, const T1* x, int incx, int stridex, int batch_count, T2* result);

template <typename T1, typename T2, typename T3 = T1>
hipblasStatus_t hipblasRot(
    hipblasHandle_t handle, int n, T1* x, int incx, T1* y, int incy, const T2* c, const T3* s);

template <typename T1, typename T2 = T1, typename T3 = T1>
hipblasStatus_t hipblasRotBatched(hipblasHandle_t handle,
                                  int             n,
                                  T1* const       x[],
                                  int             incx,
                                  T1* const       y[],
                                  int             incy,
                                  const T2*       c,
                                  const T3*       s,
                                  int             batch_count);

template <typename T1, typename T2 = T1, typename T3 = T1>
hipblasStatus_t hipblasRotStridedBatched(hipblasHandle_t handle,
                                         int             n,
                                         T1*             x,
                                         int             incx,
                                         int             stridex,
                                         T1*             y,
                                         int             incy,
                                         int             stridey,
                                         const T2*       c,
                                         const T3*       s,
                                         int             batch_count);

template <typename T1, typename T2 = T1>
hipblasStatus_t hipblasRotg(hipblasHandle_t handle, T1* a, T1* b, T2* c, T1* s);

template <typename T1, typename T2 = T1>
hipblasStatus_t hipblasRotgBatched(hipblasHandle_t handle,
                                   T1* const       a[],
                                   T1* const       b[],
                                   T2* const       c[],
                                   T1* const       s[],
                                   int             batch_count);

template <typename T1, typename T2 = T1>
hipblasStatus_t hipblasRotgStridedBatched(hipblasHandle_t handle,
                                          T1*             a,
                                          int             stridea,
                                          T1*             b,
                                          int             strideb,
                                          T2*             c,
                                          int             stridec,
                                          T1*             s,
                                          int             strides,
                                          int             batch_count);

template <typename T>
hipblasStatus_t
    hipblasRotm(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy, const T* param);

template <typename T>
hipblasStatus_t hipblasRotmBatched(hipblasHandle_t handle,
                                   int             n,
                                   T* const        x[],
                                   int             incx,
                                   T* const        y[],
                                   int             incy,
                                   const T* const  param[],
                                   int             batch_count);

template <typename T>
hipblasStatus_t hipblasRotmStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          T*              x,
                                          int             incx,
                                          int             stridex,
                                          T*              y,
                                          int             incy,
                                          int             stridey,
                                          const T*        param,
                                          int             strideparam,
                                          int             batch_count);

template <typename T>
hipblasStatus_t hipblasRotmg(hipblasHandle_t handle, T* d1, T* d2, T* x1, const T* y1, T* param);

template <typename T>
hipblasStatus_t hipblasRotmgBatched(hipblasHandle_t handle,
                                    T* const        d1[],
                                    T* const        d2[],
                                    T* const        x1[],
                                    const T* const  y1[],
                                    T* const        param[],
                                    int             batch_count);

template <typename T>
hipblasStatus_t hipblasRotmgStridedBatched(hipblasHandle_t handle,
                                           T*              d1,
                                           int             stride_d1,
                                           T*              d2,
                                           int             stride_d2,
                                           T*              x1,
                                           int             stride_x1,
                                           const T*        y1,
                                           int             stride_y1,
                                           T*              param,
                                           int             strideparam,
                                           int             batch_count);

template <typename T>
hipblasStatus_t hipblasIamax(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T>
hipblasStatus_t hipblasIamin(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T>
hipblasStatus_t hipblasAxpy(
    hipblasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy);

template <typename T>
hipblasStatus_t hipblasGer(hipblasHandle_t handle,
                           int             m,
                           int             n,
                           const T*        alpha,
                           const T*        x,
                           int             incx,
                           const T*        y,
                           int             incy,
                           T*              A,
                           int             lda);

template <typename T>
hipblasStatus_t hipblasGerBatched(hipblasHandle_t handle,
                                  int             m,
                                  int             n,
                                  const T*        alpha,
                                  const T* const  x[],
                                  int             incx,
                                  const T* const  y[],
                                  int             incy,
                                  T* const        A[],
                                  int             lda,
                                  int             batch_count);

template <typename T>
hipblasStatus_t hipblasGerStridedBatched(hipblasHandle_t handle,
                                         int             m,
                                         int             n,
                                         const T*        alpha,
                                         const T*        x,
                                         int             incx,
                                         int             stridex,
                                         const T*        y,
                                         int             incy,
                                         int             stridey,
                                         T*              A,
                                         int             lda,
                                         int             strideA,
                                         int             batch_count);

// syr
template <typename T>
hipblasStatus_t hipblasSyr(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const T*          alpha,
                           const T*          x,
                           int               incx,
                           T*                A,
                           int               lda);

template <typename T>
hipblasStatus_t hipblasSyrBatched(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T* const    x[],
                                  int               incx,
                                  T* const          A[],
                                  int               lda,
                                  int               batch_count);

template <typename T>
hipblasStatus_t hipblasSyrStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const T*          alpha,
                                         const T*          x,
                                         int               incx,
                                         int               stridex,
                                         T*                A,
                                         int               lda,
                                         int               strideA,
                                         int               batch_count);

// trsv
template <typename T>
hipblasStatus_t hipblasTrsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const T*           A,
                            int                lda,
                            T*                 x,
                            int                incx);

template <typename T>
hipblasStatus_t hipblasGemv(hipblasHandle_t    handle,
                            hipblasOperation_t transA,
                            int                m,
                            int                n,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            const T*           x,
                            int                incx,
                            const T*           beta,
                            T*                 y,
                            int                incy);

template <typename T>
hipblasStatus_t hipblasGemvBatched(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   const T* const     x[],
                                   int                incx,
                                   const T*           beta,
                                   T* const           y[],
                                   int                incy,
                                   int                batch_count);

template <typename T>
hipblasStatus_t hipblasGemvStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          const T*           x,
                                          int                incx,
                                          int                stridex,
                                          const T*           beta,
                                          T*                 y,
                                          int                incy,
                                          int                stridey,
                                          int                batch_count);

template <typename T>
hipblasStatus_t hipblasSymv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          A,
                            int               lda,
                            const T*          x,
                            int               incx,
                            const T*          beta,
                            T*                y,
                            int               incy);

template <typename T>
hipblasStatus_t hipblasGemm(hipblasHandle_t    handle,
                            hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int                m,
                            int                n,
                            int                k,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            const T*           B,
                            int                ldb,
                            const T*           beta,
                            T*                 C,
                            int                ldc);

template <typename T>
hipblasStatus_t hipblasGemmStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          int                k,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                bsa,
                                          const T*           B,
                                          int                ldb,
                                          int                bsb,
                                          const T*           beta,
                                          T*                 C,
                                          int                ldc,
                                          int                bsc,
                                          int                batch_count);

template <typename T>
hipblasStatus_t hipblasGemmBatched(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   int                k,
                                   const T*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   const T* const     B[],
                                   int                ldb,
                                   const T*           beta,
                                   T* const           C[],
                                   int                ldc,
                                   int                batch_count);

template <typename T>
hipblasStatus_t hipblasTrsm(hipblasHandle_t    handle,
                            hipblasSideMode_t  side,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            int                n,
                            const T*           alpha,
                            T*                 A,
                            int                lda,
                            T*                 B,
                            int                ldb);

template <typename T>
hipblasStatus_t hipblasTrtri(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             hipblasDiagType_t diag,
                             int               n,
                             T*                A,
                             int               lda,
                             T*                invA,
                             int               ldinvA);

template <typename T>
hipblasStatus_t hipblasTrtri_batched(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     T*                A,
                                     int               lda,
                                     int               bsa,
                                     T*                invA,
                                     int               ldinvA,
                                     int               bsinvA,
                                     int               batch_count);

template <typename T, int NB>
hipblasStatus_t hipblasTrtri_trsm(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  hipblasDiagType_t diag,
                                  int               n,
                                  T*                A,
                                  int               lda,
                                  T*                invA);

template <typename T>
hipblasStatus_t hipblasGeam(hipblasHandle_t    handle,
                            hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int                m,
                            int                n,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            const T*           beta,
                            const T*           B,
                            int                ldb,
                            T*                 C,
                            int                ldc);

#endif // _ROCBLAS_HPP_
