/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
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
template <typename T, typename U = T, bool FORTRAN = false>
hipblasStatus_t hipblasScal(hipblasHandle_t handle, int n, const U* alpha, T* x, int incx);

template <typename T, typename U = T, bool FORTRAN = false>
hipblasStatus_t hipblasScalBatched(
    hipblasHandle_t handle, int n, const U* alpha, T* const x[], int incx, int batch_count);

template <typename T, typename U = T, bool FORTRAN = false>
hipblasStatus_t hipblasScalStridedBatched(
    hipblasHandle_t handle, int n, const U* alpha, T* x, int incx, int stridex, int batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasCopy(hipblasHandle_t handle, int n, const T* x, int incx, T* y, int incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasCopyBatched(hipblasHandle_t handle,
                                   int             n,
                                   const T* const  x[],
                                   int             incx,
                                   T* const        y[],
                                   int             incy,
                                   int             batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasCopyStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const T*        x,
                                          int             incx,
                                          int             stridex,
                                          T*              y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSwap(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSwapBatched(
    hipblasHandle_t handle, int n, T* x[], int incx, T* y[], int incy, int batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSwapStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          T*              x,
                                          int             incx,
                                          int             stridex,
                                          T*              y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDot(
    hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDotc(
    hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDotBatched(hipblasHandle_t handle,
                                  int             n,
                                  const T* const  x[],
                                  int             incx,
                                  const T* const  y[],
                                  int             incy,
                                  int             batch_count,
                                  T*              result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDotcBatched(hipblasHandle_t handle,
                                   int             n,
                                   const T* const  x[],
                                   int             incx,
                                   const T* const  y[],
                                   int             incy,
                                   int             batch_count,
                                   T*              result);

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
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

template <typename T1, typename T2, bool FORTRAN = false>
hipblasStatus_t hipblasAsum(hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2, bool FORTRAN = false>
hipblasStatus_t hipblasAsumBatched(
    hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

template <typename T1, typename T2, bool FORTRAN = false>
hipblasStatus_t hipblasAsumStridedBatched(
    hipblasHandle_t handle, int n, const T1* x, int incx, int stridex, int batch_count, T2* result);

template <typename T1, typename T2, bool FORTRAN = false>
hipblasStatus_t hipblasNrm2(hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2, bool FORTRAN = false>
hipblasStatus_t hipblasNrm2Batched(
    hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

template <typename T1, typename T2, bool FORTRAN = false>
hipblasStatus_t hipblasNrm2StridedBatched(
    hipblasHandle_t handle, int n, const T1* x, int incx, int stridex, int batch_count, T2* result);

template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
hipblasStatus_t hipblasRot(
    hipblasHandle_t handle, int n, T1* x, int incx, T1* y, int incy, const T2* c, const T3* s);

template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
hipblasStatus_t hipblasRotBatched(hipblasHandle_t handle,
                                  int             n,
                                  T1* const       x[],
                                  int             incx,
                                  T1* const       y[],
                                  int             incy,
                                  const T2*       c,
                                  const T3*       s,
                                  int             batch_count);

template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
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

template <typename T1, typename T2 = T1, bool FORTRAN = false>
hipblasStatus_t hipblasRotg(hipblasHandle_t handle, T1* a, T1* b, T2* c, T1* s);

template <typename T1, typename T2 = T1, bool FORTRAN = false>
hipblasStatus_t hipblasRotgBatched(hipblasHandle_t handle,
                                   T1* const       a[],
                                   T1* const       b[],
                                   T2* const       c[],
                                   T1* const       s[],
                                   int             batch_count);

template <typename T1, typename T2 = T1, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t
    hipblasRotm(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy, const T* param);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasRotmBatched(hipblasHandle_t handle,
                                   int             n,
                                   T* const        x[],
                                   int             incx,
                                   T* const        y[],
                                   int             incy,
                                   const T* const  param[],
                                   int             batch_count);

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasRotmg(hipblasHandle_t handle, T* d1, T* d2, T* x1, const T* y1, T* param);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasRotmgBatched(hipblasHandle_t handle,
                                    T* const        d1[],
                                    T* const        d2[],
                                    T* const        x1[],
                                    const T* const  y1[],
                                    T* const        param[],
                                    int             batch_count);

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasIamax(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasIamaxBatched(
    hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasIamaxStridedBatched(
    hipblasHandle_t handle, int n, const T* x, int incx, int stridex, int batch_count, int* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasIamin(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasIaminBatched(
    hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasIaminStridedBatched(
    hipblasHandle_t handle, int n, const T* x, int incx, int stridex, int batch_count, int* result);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasAxpy(
    hipblasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasAxpyBatched(hipblasHandle_t handle,
                                   int             n,
                                   const T*        alpha,
                                   const T* const  x[],
                                   int             incx,
                                   T* const        y[],
                                   int             incy,
                                   int             batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasAxpyStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const T*        alpha,
                                          const T*        x,
                                          int             incx,
                                          int             stridex,
                                          T*              y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count);

// ger
template <typename T, bool CONJ, bool FORTRAN = false>
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

template <typename T, bool CONJ, bool FORTRAN = false>
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

template <typename T, bool CONJ, bool FORTRAN = false>
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

// hbmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHbmv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            int               k,
                            const T*          alpha,
                            const T*          A,
                            int               lda,
                            const T*          x,
                            int               incx,
                            const T*          beta,
                            T*                y,
                            int               incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHbmvBatched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   int               k,
                                   const T*          alpha,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    x[],
                                   int               incx,
                                   const T*          beta,
                                   T* const          y[],
                                   int               incy,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHbmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          int               strideA,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount);

// hemv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHemv(hipblasHandle_t   handle,
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHemvBatched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    x[],
                                   int               incx,
                                   const T*          beta,
                                   T* const          y[],
                                   int               incy,
                                   int               batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHemvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          int               stride_a,
                                          const T*          x,
                                          int               incx,
                                          int               stride_x,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          int               stride_y,
                                          int               batch_count);

// her
template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHer(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const U*          alpha,
                           const T*          x,
                           int               incx,
                           T*                A,
                           int               lda);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerBatched(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const U*          alpha,
                                  const T* const    x[],
                                  int               incx,
                                  T* const          A[],
                                  int               lda,
                                  int               batchCount);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const U*          alpha,
                                         const T*          x,
                                         int               incx,
                                         int               stridex,
                                         T*                A,
                                         int               lda,
                                         int               strideA,
                                         int               batchCount);

// her2
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHer2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          x,
                            int               incx,
                            const T*          y,
                            int               incy,
                            T*                A,
                            int               lda);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHer2Batched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    x[],
                                   int               incx,
                                   const T* const    y[],
                                   int               incy,
                                   T* const          A[],
                                   int               lda,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHer2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          y,
                                          int               incy,
                                          int               stridey,
                                          T*                A,
                                          int               lda,
                                          int               strideA,
                                          int               batchCount);

// hpmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHpmv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          AP,
                            const T*          x,
                            int               incx,
                            const T*          beta,
                            T*                y,
                            int               incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHpmvBatched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    AP[],
                                   const T* const    x[],
                                   int               incx,
                                   const T*          beta,
                                   T* const          y[],
                                   int               incy,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHpmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          AP,
                                          int               strideAP,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount);

// hpr
template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHpr(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const U*          alpha,
                           const T*          x,
                           int               incx,
                           T*                AP);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHprBatched(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const U*          alpha,
                                  const T* const    x[],
                                  int               incx,
                                  T* const          AP[],
                                  int               batchCount);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHprStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const U*          alpha,
                                         const T*          x,
                                         int               incx,
                                         int               stridex,
                                         T*                AP,
                                         int               strideAP,
                                         int               batchCount);

// hpr2
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHpr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          x,
                            int               incx,
                            const T*          y,
                            int               incy,
                            T*                AP);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHpr2Batched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    x[],
                                   int               incx,
                                   const T* const    y[],
                                   int               incy,
                                   T* const          AP[],
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHpr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          y,
                                          int               incy,
                                          int               stridey,
                                          T*                AP,
                                          int               strideAP,
                                          int               batchCount);

// sbmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSbmv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            int               k,
                            const T*          alpha,
                            const T*          A,
                            int               lda,
                            const T*          x,
                            int               incx,
                            const T*          beta,
                            T*                y,
                            int               incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSbmvBatched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   int               k,
                                   const T*          alpha,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    x[],
                                   int               incx,
                                   const T*          beta,
                                   T*                y[],
                                   int               incy,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSbmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          int               strideA,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount);

// spmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpmv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          AP,
                            const T*          x,
                            int               incx,
                            const T*          beta,
                            T*                y,
                            int               incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpmvBatched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    AP[],
                                   const T* const    x[],
                                   int               incx,
                                   const T*          beta,
                                   T*                y[],
                                   int               incy,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          AP,
                                          int               strideAP,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount);

// spr
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpr(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const T*          alpha,
                           const T*          x,
                           int               incx,
                           T*                AP);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSprBatched(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T* const    x[],
                                  int               incx,
                                  T* const          AP[],
                                  int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSprStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const T*          alpha,
                                         const T*          x,
                                         int               incx,
                                         int               stridex,
                                         T*                AP,
                                         int               strideAP,
                                         int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          x,
                            int               incx,
                            const T*          y,
                            int               incy,
                            T*                AP);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpr2Batched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    x[],
                                   int               incx,
                                   const T* const    y[],
                                   int               incy,
                                   T* const          AP[],
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSpr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          y,
                                          int               incy,
                                          int               stridey,
                                          T*                AP,
                                          int               strideAP,
                                          int               batchCount);

// symv
template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSymvBatched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    x[],
                                   int               incx,
                                   const T*          beta,
                                   T*                y[],
                                   int               incy,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSymvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          int               strideA,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount);

// syr
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const T*          alpha,
                           const T*          x,
                           int               incx,
                           T*                A,
                           int               lda);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrBatched(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T* const    x[],
                                  int               incx,
                                  T* const          A[],
                                  int               lda,
                                  int               batch_count);

template <typename T, bool FORTRAN = false>
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

// syr2
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const T*          alpha,
                            const T*          x,
                            int               incx,
                            const T*          y,
                            int               incy,
                            T*                A,
                            int               lda);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr2Batched(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    x[],
                                   int               incx,
                                   const T* const    y[],
                                   int               incy,
                                   T* const          A[],
                                   int               lda,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          x,
                                          int               incx,
                                          int               stridex,
                                          const T*          y,
                                          int               incy,
                                          int               stridey,
                                          T*                A,
                                          int               lda,
                                          int               strideA,
                                          int               batchCount);

// tbmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTbmv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            int                k,
                            const T*           A,
                            int                lda,
                            T*                 x,
                            int                incx);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTbmvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                k,
                                   const T* const     A[],
                                   int                lda,
                                   T* const           x[],
                                   int                incx,
                                   int                batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTbmvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                k,
                                          const T*           A,
                                          int                lda,
                                          int                stride_a,
                                          T*                 x,
                                          int                incx,
                                          int                stride_x,
                                          int                batch_count);

// tbsv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTbsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            int                k,
                            const T*           A,
                            int                lda,
                            T*                 x,
                            int                incx);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTbsvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                k,
                                   const T* const     A[],
                                   int                lda,
                                   T* const           x[],
                                   int                incx,
                                   int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTbsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                k,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          T*                 x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount);

// tpmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTpmv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const T*           AP,
                            T*                 x,
                            int                incx);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTpmvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T* const     AP[],
                                   T* const           x[],
                                   int                incx,
                                   int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTpmvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T*           AP,
                                          int                strideAP,
                                          T*                 x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount);

// tpsv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTpsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const T*           AP,
                            T*                 x,
                            int                incx);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTpsvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T* const     AP[],
                                   T* const           x[],
                                   int                incx,
                                   int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTpsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T*           AP,
                                          int                strideAP,
                                          T*                 x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount);

// trmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrmv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const T*           A,
                            int                lda,
                            T*                 x,
                            int                incx);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrmvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T* const     A[],
                                   int                lda,
                                   T* const           x[],
                                   int                incx,
                                   int                batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrmvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T*           A,
                                          int                lda,
                                          int                stride_a,
                                          T*                 x,
                                          int                incx,
                                          int                stride_x,
                                          int                batch_count);

// trsv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const T*           A,
                            int                lda,
                            T*                 x,
                            int                incx);

// trsv_batched
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrsvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T* const     A[],
                                   int                lda,
                                   T* const           x[],
                                   int                incx,
                                   int                batch_count);

// trsv_strided_batched
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          T*                 x,
                                          int                incx,
                                          int                stridex,
                                          int                batch_count);

// gbmv
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGbmv(hipblasHandle_t    handle,
                            hipblasOperation_t transA,
                            int                m,
                            int                n,
                            int                kl,
                            int                ku,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            const T*           x,
                            int                incx,
                            const T*           beta,
                            T*                 y,
                            int                incy);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGbmvBatched(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   int                m,
                                   int                n,
                                   int                kl,
                                   int                ku,
                                   const T*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   const T* const     x[],
                                   int                incx,
                                   const T*           beta,
                                   T* const           y[],
                                   int                incy,
                                   int                batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGbmvStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          int                m,
                                          int                n,
                                          int                kl,
                                          int                ku,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                stride_a,
                                          const T*           x,
                                          int                incx,
                                          int                stride_x,
                                          const T*           beta,
                                          T*                 y,
                                          int                incy,
                                          int                stride_y,
                                          int                batch_count);

// gemv
template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
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

// herk
template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerk(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            int                n,
                            int                k,
                            const U*           alpha,
                            const T*           A,
                            int                lda,
                            const U*           beta,
                            T*                 C,
                            int                ldc);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerkBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   int                n,
                                   int                k,
                                   const U*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   const U*           beta,
                                   T* const           C[],
                                   int                ldc,
                                   int                batchCount);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerkStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          int                n,
                                          int                k,
                                          const U*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          const U*           beta,
                                          T*                 C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount);

// her2k
template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHer2k(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const T*           alpha,
                             const T*           A,
                             int                lda,
                             const T*           B,
                             int                ldb,
                             const U*           beta,
                             T*                 C,
                             int                ldc);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHer2kBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const T*           alpha,
                                    const T* const     A[],
                                    int                lda,
                                    const T* const     B[],
                                    int                ldb,
                                    const U*           beta,
                                    T* const           C[],
                                    int                ldc,
                                    int                batchCount);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHer2kStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T*           A,
                                           int                lda,
                                           int                strideA,
                                           const T*           B,
                                           int                ldb,
                                           int                strideB,
                                           const U*           beta,
                                           T*                 C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount);

// herkx
template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerkx(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const T*           alpha,
                             const T*           A,
                             int                lda,
                             const T*           B,
                             int                ldb,
                             const U*           beta,
                             T*                 C,
                             int                ldc);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerkxBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const T*           alpha,
                                    const T* const     A[],
                                    int                lda,
                                    const T* const     B[],
                                    int                ldb,
                                    const U*           beta,
                                    T* const           C[],
                                    int                ldc,
                                    int                batchCount);

template <typename T, typename U, bool FORTRAN = false>
hipblasStatus_t hipblasHerkxStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T*           A,
                                           int                lda,
                                           int                strideA,
                                           const T*           B,
                                           int                ldb,
                                           int                strideB,
                                           const U*           beta,
                                           T*                 C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount);

// symm
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSymm(hipblasHandle_t   handle,
                            hipblasSideMode_t side,
                            hipblasFillMode_t uplo,
                            int               m,
                            int               n,
                            const T*          alpha,
                            const T*          A,
                            int               lda,
                            const T*          B,
                            int               ldb,
                            const T*          beta,
                            T*                C,
                            int               ldc);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSymmBatched(hipblasHandle_t   handle,
                                   hipblasSideMode_t side,
                                   hipblasFillMode_t uplo,
                                   int               m,
                                   int               n,
                                   const T*          alpha,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    B[],
                                   int               ldb,
                                   const T*          beta,
                                   T* const          C[],
                                   int               ldc,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSymmStridedBatched(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          hipblasFillMode_t uplo,
                                          int               m,
                                          int               n,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          int               strideA,
                                          const T*          B,
                                          int               ldb,
                                          int               strideB,
                                          const T*          beta,
                                          T*                C,
                                          int               ldc,
                                          int               strideC,
                                          int               batchCount);

// syrk
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrk(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            int                n,
                            int                k,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            const T*           beta,
                            T*                 C,
                            int                ldc);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrkBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   int                n,
                                   int                k,
                                   const T*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   const T*           beta,
                                   T* const           C[],
                                   int                ldc,
                                   int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrkStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          int                n,
                                          int                k,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          const T*           beta,
                                          T*                 C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount);

// syr2k
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr2k(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr2kBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
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
                                    int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyr2kStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T*           A,
                                           int                lda,
                                           int                strideA,
                                           const T*           B,
                                           int                ldb,
                                           int                strideB,
                                           const T*           beta,
                                           T*                 C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount);

// syrkx
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrkx(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrkxBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
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
                                    int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSyrkxStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T*           A,
                                           int                lda,
                                           int                strideA,
                                           const T*           B,
                                           int                ldb,
                                           int                strideB,
                                           const T*           beta,
                                           T*                 C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount);

// geam
template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGeamBatched(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   const T*           beta,
                                   const T* const     B[],
                                   int                ldb,
                                   T* const           C[],
                                   int                ldc,
                                   int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGeamStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          const T*           beta,
                                          const T*           B,
                                          int                ldb,
                                          int                strideB,
                                          T*                 C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount);

// hemm
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHemm(hipblasHandle_t   handle,
                            hipblasSideMode_t side,
                            hipblasFillMode_t uplo,
                            int               n,
                            int               k,
                            const T*          alpha,
                            const T*          A,
                            int               lda,
                            const T*          B,
                            int               ldb,
                            const T*          beta,
                            T*                C,
                            int               ldc);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHemmBatched(hipblasHandle_t   handle,
                                   hipblasSideMode_t side,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   int               k,
                                   const T*          alpha,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    B[],
                                   int               ldb,
                                   const T*          beta,
                                   T* const          C[],
                                   int               ldc,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasHemmStridedBatched(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          int               strideA,
                                          const T*          B,
                                          int               ldb,
                                          int               strideB,
                                          const T*          beta,
                                          T*                C,
                                          int               ldc,
                                          int               strideC,
                                          int               batchCount);

// trmm
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrmm(hipblasHandle_t    handle,
                            hipblasSideMode_t  side,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            int                n,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            T*                 B,
                            int                ldb);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrmmBatched(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T* const     A[],
                                   int                lda,
                                   T* const           B[],
                                   int                ldb,
                                   int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrmmStridedBatched(hipblasHandle_t    handle,
                                          hipblasSideMode_t  side,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                strideA,
                                          T*                 B,
                                          int                ldb,
                                          int                strideB,
                                          int                batchCount);

// trsm
template <typename T, bool FORTRAN = false>
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

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrsmBatched(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   T* const           A[],
                                   int                lda,
                                   T*                 B[],
                                   int                ldb,
                                   int                batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrsmStridedBatched(hipblasHandle_t    handle,
                                          hipblasSideMode_t  side,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          T*                 A,
                                          int                lda,
                                          int                strideA,
                                          T*                 B,
                                          int                ldb,
                                          int                strideB,
                                          int                batch_count);

// getrf
template <typename T, bool FORTRAN = false>
hipblasStatus_t
    hipblasGetrf(hipblasHandle_t handle, const int n, T* A, const int lda, int* ipiv, int* info);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGetrfBatched(hipblasHandle_t handle,
                                    const int       n,
                                    T* const        A[],
                                    const int       lda,
                                    int*            ipiv,
                                    int*            info,
                                    const int       batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGetrfStridedBatched(hipblasHandle_t handle,
                                           const int       n,
                                           T*              A,
                                           const int       lda,
                                           const int       strideA,
                                           int*            ipiv,
                                           const int       strideP,
                                           int*            info,
                                           const int       batchCount);

// getrs
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGetrs(hipblasHandle_t          handle,
                             const hipblasOperation_t trans,
                             const int                n,
                             const int                nrhs,
                             T*                       A,
                             const int                lda,
                             const int*               ipiv,
                             T*                       B,
                             const int                ldb,
                             int*                     info);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGetrsBatched(hipblasHandle_t          handle,
                                    const hipblasOperation_t trans,
                                    const int                n,
                                    const int                nrhs,
                                    T* const                 A[],
                                    const int                lda,
                                    const int*               ipiv,
                                    T* const                 B[],
                                    const int                ldb,
                                    int*                     info,
                                    const int                batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGetrsStridedBatched(hipblasHandle_t          handle,
                                           const hipblasOperation_t trans,
                                           const int                n,
                                           const int                nrhs,
                                           T*                       A,
                                           const int                lda,
                                           const int                strideA,
                                           const int*               ipiv,
                                           const int                strideP,
                                           T*                       B,
                                           const int                ldb,
                                           const int                strideB,
                                           int*                     info,
                                           const int                batchCount);

// getri
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGetriBatched(hipblasHandle_t handle,
                                    const int       n,
                                    T* const        A[],
                                    const int       lda,
                                    int*            ipiv,
                                    T* const        C[],
                                    const int       ldc,
                                    int*            info,
                                    const int       batchCount);

// geqrf
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGeqrf(
    hipblasHandle_t handle, const int m, const int n, T* A, const int lda, T* ipiv, int* info);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGeqrfBatched(hipblasHandle_t handle,
                                    const int       m,
                                    const int       n,
                                    T* const        A[],
                                    const int       lda,
                                    T* const        ipiv[],
                                    int*            info,
                                    const int       batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGeqrfStridedBatched(hipblasHandle_t handle,
                                           const int       m,
                                           const int       n,
                                           T*              A,
                                           const int       lda,
                                           const int       strideA,
                                           T*              ipiv,
                                           const int       strideP,
                                           int*            info,
                                           const int       batchCount);

// dgmm
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDgmm(hipblasHandle_t   handle,
                            hipblasSideMode_t side,
                            int               m,
                            int               n,
                            const T*          A,
                            int               lda,
                            const T*          x,
                            int               incx,
                            T*                C,
                            int               ldc);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDgmmBatched(hipblasHandle_t   handle,
                                   hipblasSideMode_t side,
                                   int               m,
                                   int               n,
                                   const T* const    A[],
                                   int               lda,
                                   const T* const    x[],
                                   int               incx,
                                   T* const          C[],
                                   int               ldc,
                                   int               batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasDgmmStridedBatched(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          int               m,
                                          int               n,
                                          const T*          A,
                                          int               lda,
                                          int               stride_A,
                                          const T*          x,
                                          int               incx,
                                          int               stride_x,
                                          T*                C,
                                          int               ldc,
                                          int               stride_C,
                                          int               batch_count);

// trtri
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrtri(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             hipblasDiagType_t diag,
                             int               n,
                             T*                A,
                             int               lda,
                             T*                invA,
                             int               ldinvA);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrtriBatched(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    hipblasDiagType_t diag,
                                    int               n,
                                    T*                A[],
                                    int               lda,
                                    T*                invA[],
                                    int               ldinvA,
                                    int               batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrtriStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           hipblasDiagType_t diag,
                                           int               n,
                                           T*                A,
                                           int               lda,
                                           int               stride_A,
                                           T*                invA,
                                           int               ldinvA,
                                           int               stride_invA,
                                           int               batch_count);

template <typename T, int NB>
hipblasStatus_t hipblasTrtri_trsm(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  hipblasDiagType_t diag,
                                  int               n,
                                  T*                A,
                                  int               lda,
                                  T*                invA);

#endif // _ROCBLAS_HPP_
