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
template <typename T>
hipblasStatus_t hipblasScal(hipblasHandle_t handle, int n, const T* alpha, T* x, int incx);

template <typename T>
hipblasStatus_t hipblasCopy(hipblasHandle_t handle, int n, const T* x, int incx, T* y, int incy);

template <typename T>
hipblasStatus_t hipblasSwap(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy);

template <typename T>
hipblasStatus_t hipblasDot(
    hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasAsum(hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

template <typename T1, typename T2>
hipblasStatus_t hipblasNrm2(hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

template <typename T>
hipblasStatus_t hipblasIamax(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T>
hipblasStatus_t hipblasAmin(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

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
                                   const T*           A[],
                                   int                lda,
                                   const T*           B[],
                                   int                ldb,
                                   const T*           beta,
                                   T*                 C[],
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
