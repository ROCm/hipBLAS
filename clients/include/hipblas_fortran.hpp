/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _HIPBLAS_FORTRAN_HPP
#define _HIPBLAS_FORTRAN_HPP

/*!\file
 *  This file interfaces with our Fortran BLAS interface.
 */

/*
 * ============================================================================
 *     Fortran functions
 * ============================================================================
 */

// Temporarily replacing hipblasComplex with hipComplex
// #ifdef HIPBLAS_V2
// #define hipblasComplex hipComplex
// #define hipblasDoubleComplex hipDoubleComplex
// #endif
extern "C" {

/* ==========
 *    Aux
 * ========== */
hipblasStatus_t
    hipblasSetVectorFortran(int n, int elemSize, const void* x, int incx, void* y, int incy);

hipblasStatus_t
    hipblasGetVectorFortran(int n, int elemSize, const void* x, int incx, void* y, int incy);

hipblasStatus_t hipblasSetMatrixFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

hipblasStatus_t hipblasGetMatrixFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

hipblasStatus_t hipblasSetVectorAsyncFortran(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream);

hipblasStatus_t hipblasGetVectorAsyncFortran(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream);

hipblasStatus_t hipblasSetMatrixAsyncFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream);

hipblasStatus_t hipblasGetMatrixAsyncFortran(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream);

hipblasStatus_t hipblasSetAtomicsModeFortran(hipblasHandle_t      handle,
                                             hipblasAtomicsMode_t atomics_mode);

hipblasStatus_t hipblasGetAtomicsModeFortran(hipblasHandle_t       handle,
                                             hipblasAtomicsMode_t* atomics_mode);

/* ==========
 *    L1
 * ========== */

// scal
hipblasStatus_t
    hipblasSscalFortran(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx);

hipblasStatus_t
    hipblasDscalFortran(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCscalFortran(
    hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx);

hipblasStatus_t hipblasZscalFortran(
    hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx);

hipblasStatus_t hipblasCsscalFortran(
    hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx);

hipblasStatus_t hipblasZdscalFortran(
    hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx);
#else
hipblasStatus_t hipblasCscalFortran(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx);

hipblasStatus_t hipblasZscalFortran(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);

hipblasStatus_t hipblasCsscalFortran(
    hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx);

hipblasStatus_t hipblasZdscalFortran(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx);
#endif

// scalBatched
hipblasStatus_t hipblasSscalBatchedFortran(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batch_count);

hipblasStatus_t hipblasDscalBatchedFortran(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double* const   x[],
                                           int             incx,
                                           int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCscalBatchedFortran(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* alpha,
                                           hipComplex* const x[],
                                           int               incx,
                                           int               batch_count);

hipblasStatus_t hipblasZscalBatchedFortran(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* alpha,
                                           hipDoubleComplex* const x[],
                                           int                     incx,
                                           int                     batch_count);

hipblasStatus_t hipblasCsscalBatchedFortran(hipblasHandle_t   handle,
                                            int               n,
                                            const float*      alpha,
                                            hipComplex* const x[],
                                            int               incx,
                                            int               batch_count);

hipblasStatus_t hipblasZdscalBatchedFortran(hipblasHandle_t         handle,
                                            int                     n,
                                            const double*           alpha,
                                            hipDoubleComplex* const x[],
                                            int                     incx,
                                            int                     batch_count);
#else
hipblasStatus_t hipblasCscalBatchedFortran(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex* const x[],
                                           int                   incx,
                                           int                   batch_count);

hipblasStatus_t hipblasZscalBatchedFortran(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex* const x[],
                                           int                         incx,
                                           int                         batch_count);

hipblasStatus_t hipblasCsscalBatchedFortran(hipblasHandle_t       handle,
                                            int                   n,
                                            const float*          alpha,
                                            hipblasComplex* const x[],
                                            int                   incx,
                                            int                   batch_count);

hipblasStatus_t hipblasZdscalBatchedFortran(hipblasHandle_t             handle,
                                            int                         n,
                                            const double*               alpha,
                                            hipblasDoubleComplex* const x[],
                                            int                         incx,
                                            int                         batch_count);
#endif

// scalStridedBatched
hipblasStatus_t hipblasSscalStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const float*    alpha,
                                                  float*          x,
                                                  int             incx,
                                                  hipblasStride   stride_x,
                                                  int             batch_count);

hipblasStatus_t hipblasDscalStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   alpha,
                                                  double*         x,
                                                  int             incx,
                                                  hipblasStride   stride_x,
                                                  int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCscalStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  hipComplex*       x,
                                                  int               incx,
                                                  hipblasStride     stride_x,
                                                  int               batch_count);

hipblasStatus_t hipblasZscalStridedBatchedFortran(hipblasHandle_t         handle,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stride_x,
                                                  int                     batch_count);

hipblasStatus_t hipblasCsscalStridedBatchedFortran(hipblasHandle_t handle,
                                                   int             n,
                                                   const float*    alpha,
                                                   hipComplex*     x,
                                                   int             incx,
                                                   hipblasStride   stride_x,
                                                   int             batch_count);

hipblasStatus_t hipblasZdscalStridedBatchedFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const double*     alpha,
                                                   hipDoubleComplex* x,
                                                   int               incx,
                                                   hipblasStride     stride_x,
                                                   int               batch_count);
#else
hipblasStatus_t hipblasCscalStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  int                   batch_count);

hipblasStatus_t hipblasZscalStridedBatchedFortran(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stride_x,
                                                  int                         batch_count);

hipblasStatus_t hipblasCsscalStridedBatchedFortran(hipblasHandle_t handle,
                                                   int             n,
                                                   const float*    alpha,
                                                   hipblasComplex* x,
                                                   int             incx,
                                                   hipblasStride   stride_x,
                                                   int             batch_count);

hipblasStatus_t hipblasZdscalStridedBatchedFortran(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const double*         alpha,
                                                   hipblasDoubleComplex* x,
                                                   int                   incx,
                                                   hipblasStride         stride_x,
                                                   int                   batch_count);
#endif

// copy
hipblasStatus_t hipblasScopyFortran(
    hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);

hipblasStatus_t hipblasDcopyFortran(
    hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCcopyFortran(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, hipComplex* y, int incy);

hipblasStatus_t hipblasZcopyFortran(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasCcopyFortran(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);

hipblasStatus_t hipblasZcopyFortran(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// copyBatched
hipblasStatus_t hipblasScopyBatchedFortran(hipblasHandle_t    handle,
                                           int                n,
                                           const float* const x[],
                                           int                incx,
                                           float* const       y[],
                                           int                incy,
                                           int                batch_count);

hipblasStatus_t hipblasDcopyBatchedFortran(hipblasHandle_t     handle,
                                           int                 n,
                                           const double* const x[],
                                           int                 incx,
                                           double* const       y[],
                                           int                 incy,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCcopyBatchedFortran(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batch_count);

hipblasStatus_t hipblasZcopyBatchedFortran(hipblasHandle_t               handle,
                                           int                           n,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCcopyBatchedFortran(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batch_count);

hipblasStatus_t hipblasZcopyBatchedFortran(hipblasHandle_t                   handle,
                                           int                               n,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batch_count);
#endif

// copyStridedBatched
hipblasStatus_t hipblasScopyStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const float*    x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  float*          y,
                                                  int             incy,
                                                  hipblasStride   stridey,
                                                  int             batch_count);

hipblasStatus_t hipblasDcopyStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  double*         y,
                                                  int             incy,
                                                  hipblasStride   stridey,
                                                  int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCcopyStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  hipComplex*       y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batch_count);

hipblasStatus_t hipblasZcopyStridedBatchedFortran(hipblasHandle_t         handle,
                                                  int                     n,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCcopyStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   n,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batch_count);

hipblasStatus_t hipblasZcopyStridedBatchedFortran(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batch_count);
#endif

// dot
hipblasStatus_t hipblasSdotFortran(hipblasHandle_t handle,
                                   int             n,
                                   const float*    x,
                                   int             incx,
                                   const float*    y,
                                   int             incy,
                                   float*          result);

hipblasStatus_t hipblasDdotFortran(hipblasHandle_t handle,
                                   int             n,
                                   const double*   x,
                                   int             incx,
                                   const double*   y,
                                   int             incy,
                                   double*         result);

hipblasStatus_t hipblasHdotFortran(hipblasHandle_t    handle,
                                   int                n,
                                   const hipblasHalf* x,
                                   int                incx,
                                   const hipblasHalf* y,
                                   int                incy,
                                   hipblasHalf*       result);

hipblasStatus_t hipblasBfdotFortran(hipblasHandle_t        handle,
                                    int                    n,
                                    const hipblasBfloat16* x,
                                    int                    incx,
                                    const hipblasBfloat16* y,
                                    int                    incy,
                                    hipblasBfloat16*       result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCdotuFortran(hipblasHandle_t   handle,
                                    int               n,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       result);

hipblasStatus_t hipblasZdotuFortran(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       result);

hipblasStatus_t hipblasCdotcFortran(hipblasHandle_t   handle,
                                    int               n,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       result);

hipblasStatus_t hipblasZdotcFortran(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       result);
#else
hipblasStatus_t hipblasCdotuFortran(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       result);

hipblasStatus_t hipblasZdotuFortran(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       result);

hipblasStatus_t hipblasCdotcFortran(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       result);

hipblasStatus_t hipblasZdotcFortran(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       result);
#endif

// dotBatched
hipblasStatus_t hipblasSdotBatchedFortran(hipblasHandle_t    handle,
                                          int                n,
                                          const float* const x[],
                                          int                incx,
                                          const float* const y[],
                                          int                incy,
                                          int                batch_count,
                                          float*             result);

hipblasStatus_t hipblasDdotBatchedFortran(hipblasHandle_t     handle,
                                          int                 n,
                                          const double* const x[],
                                          int                 incx,
                                          const double* const y[],
                                          int                 incy,
                                          int                 batch_count,
                                          double*             result);

hipblasStatus_t hipblasHdotBatchedFortran(hipblasHandle_t          handle,
                                          int                      n,
                                          const hipblasHalf* const x[],
                                          int                      incx,
                                          const hipblasHalf* const y[],
                                          int                      incy,
                                          int                      batch_count,
                                          hipblasHalf*             result);

hipblasStatus_t hipblasBfdotBatchedFortran(hipblasHandle_t              handle,
                                           int                          n,
                                           const hipblasBfloat16* const x[],
                                           int                          incx,
                                           const hipblasBfloat16* const y[],
                                           int                          incy,
                                           int                          batch_count,
                                           hipblasBfloat16*             result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCdotuBatchedFortran(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           int                     batch_count,
                                           hipComplex*             result);

hipblasStatus_t hipblasZdotuBatchedFortran(hipblasHandle_t               handle,
                                           int                           n,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           int                           batch_count,
                                           hipDoubleComplex*             result);

hipblasStatus_t hipblasCdotcBatchedFortran(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           int                     batch_count,
                                           hipComplex*             result);

hipblasStatus_t hipblasZdotcBatchedFortran(hipblasHandle_t               handle,
                                           int                           n,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           int                           batch_count,
                                           hipDoubleComplex*             result);
#else
hipblasStatus_t hipblasCdotuBatchedFortran(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           int                         batch_count,
                                           hipblasComplex*             result);

hipblasStatus_t hipblasZdotuBatchedFortran(hipblasHandle_t                   handle,
                                           int                               n,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           int                               batch_count,
                                           hipblasDoubleComplex*             result);

hipblasStatus_t hipblasCdotcBatchedFortran(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           int                         batch_count,
                                           hipblasComplex*             result);

hipblasStatus_t hipblasZdotcBatchedFortran(hipblasHandle_t                   handle,
                                           int                               n,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           int                               batch_count,
                                           hipblasDoubleComplex*             result);
#endif

// dotStridedBatched
hipblasStatus_t hipblasSdotStridedBatchedFortran(hipblasHandle_t handle,
                                                 int             n,
                                                 const float*    x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 const float*    y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count,
                                                 float*          result);

hipblasStatus_t hipblasDdotStridedBatchedFortran(hipblasHandle_t handle,
                                                 int             n,
                                                 const double*   x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 const double*   y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count,
                                                 double*         result);

hipblasStatus_t hipblasHdotStridedBatchedFortran(hipblasHandle_t    handle,
                                                 int                n,
                                                 const hipblasHalf* x,
                                                 int                incx,
                                                 hipblasStride      stridex,
                                                 const hipblasHalf* y,
                                                 int                incy,
                                                 hipblasStride      stridey,
                                                 int                batch_count,
                                                 hipblasHalf*       result);

hipblasStatus_t hipblasBfdotStridedBatchedFortran(hipblasHandle_t        handle,
                                                  int                    n,
                                                  const hipblasBfloat16* x,
                                                  int                    incx,
                                                  hipblasStride          stridex,
                                                  const hipblasBfloat16* y,
                                                  int                    incy,
                                                  hipblasStride          stridey,
                                                  int                    batch_count,
                                                  hipblasBfloat16*       result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCdotuStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batch_count,
                                                  hipComplex*       result);

hipblasStatus_t hipblasZdotuStridedBatchedFortran(hipblasHandle_t         handle,
                                                  int                     n,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batch_count,
                                                  hipDoubleComplex*       result);

hipblasStatus_t hipblasCdotcStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batch_count,
                                                  hipComplex*       result);

hipblasStatus_t hipblasZdotcStridedBatchedFortran(hipblasHandle_t         handle,
                                                  int                     n,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batch_count,
                                                  hipDoubleComplex*       result);
#else
hipblasStatus_t hipblasCdotuStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   n,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batch_count,
                                                  hipblasComplex*       result);

hipblasStatus_t hipblasZdotuStridedBatchedFortran(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batch_count,
                                                  hipblasDoubleComplex*       result);

hipblasStatus_t hipblasCdotcStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   n,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batch_count,
                                                  hipblasComplex*       result);

hipblasStatus_t hipblasZdotcStridedBatchedFortran(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batch_count,
                                                  hipblasDoubleComplex*       result);
#endif

// swap
hipblasStatus_t
    hipblasSswapFortran(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy);

hipblasStatus_t
    hipblasDswapFortran(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCswapFortran(
    hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy);

hipblasStatus_t hipblasZswapFortran(
    hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy);
#else
hipblasStatus_t hipblasCswapFortran(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy);

hipblasStatus_t hipblasZswapFortran(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasDoubleComplex* x,
                                    int                   incx,
                                    hipblasDoubleComplex* y,
                                    int                   incy);
#endif

// swapBatched
hipblasStatus_t hipblasSswapBatchedFortran(hipblasHandle_t handle,
                                           int             n,
                                           float* const    x[],
                                           int             incx,
                                           float* const    y[],
                                           int             incy,
                                           int             batch_count);

hipblasStatus_t hipblasDswapBatchedFortran(hipblasHandle_t handle,
                                           int             n,
                                           double* const   x[],
                                           int             incx,
                                           double* const   y[],
                                           int             incy,
                                           int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCswapBatchedFortran(hipblasHandle_t   handle,
                                           int               n,
                                           hipComplex* const x[],
                                           int               incx,
                                           hipComplex* const y[],
                                           int               incy,
                                           int               batch_count);

hipblasStatus_t hipblasZswapBatchedFortran(hipblasHandle_t         handle,
                                           int                     n,
                                           hipDoubleComplex* const x[],
                                           int                     incx,
                                           hipDoubleComplex* const y[],
                                           int                     incy,
                                           int                     batch_count);
#else
hipblasStatus_t hipblasCswapBatchedFortran(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasComplex* const x[],
                                           int                   incx,
                                           hipblasComplex* const y[],
                                           int                   incy,
                                           int                   batch_count);

hipblasStatus_t hipblasZswapBatchedFortran(hipblasHandle_t             handle,
                                           int                         n,
                                           hipblasDoubleComplex* const x[],
                                           int                         incx,
                                           hipblasDoubleComplex* const y[],
                                           int                         incy,
                                           int                         batch_count);
#endif

// swapStridedBatched
hipblasStatus_t hipblasSswapStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  float*          x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  float*          y,
                                                  int             incy,
                                                  hipblasStride   stridey,
                                                  int             batch_count);

hipblasStatus_t hipblasDswapStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  double*         x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  double*         y,
                                                  int             incy,
                                                  hipblasStride   stridey,
                                                  int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCswapStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  hipComplex*     x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  hipComplex*     y,
                                                  int             incy,
                                                  hipblasStride   stridey,
                                                  int             batch_count);

hipblasStatus_t hipblasZswapStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  hipDoubleComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  hipDoubleComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batch_count);
#else
hipblasStatus_t hipblasCswapStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  hipblasComplex* x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  hipblasComplex* y,
                                                  int             incy,
                                                  hipblasStride   stridey,
                                                  int             batch_count);

hipblasStatus_t hipblasZswapStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   n,
                                                  hipblasDoubleComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  hipblasDoubleComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batch_count);
#endif

// axpy
hipblasStatus_t hipblasHaxpyFortran(hipblasHandle_t    handle,
                                    const int          N,
                                    const hipblasHalf* alpha,
                                    const hipblasHalf* x,
                                    const int          incx,
                                    hipblasHalf*       y,
                                    const int          incy);

hipblasStatus_t hipblasSaxpyFortran(hipblasHandle_t handle,
                                    const int       N,
                                    const float*    alpha,
                                    const float*    x,
                                    const int       incx,
                                    float*          y,
                                    const int       incy);

hipblasStatus_t hipblasDaxpyFortran(hipblasHandle_t handle,
                                    const int       N,
                                    const double*   alpha,
                                    const double*   x,
                                    const int       incx,
                                    double*         y,
                                    const int       incy);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCaxpyFortran(hipblasHandle_t   handle,
                                    const int         N,
                                    const hipComplex* alpha,
                                    const hipComplex* x,
                                    const int         incx,
                                    hipComplex*       y,
                                    const int         incy);

hipblasStatus_t hipblasZaxpyFortran(hipblasHandle_t         handle,
                                    const int               N,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* x,
                                    const int               incx,
                                    hipDoubleComplex*       y,
                                    const int               incy);
#else
hipblasStatus_t hipblasCaxpyFortran(hipblasHandle_t       handle,
                                    const int             N,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    const int             incx,
                                    hipblasComplex*       y,
                                    const int             incy);

hipblasStatus_t hipblasZaxpyFortran(hipblasHandle_t             handle,
                                    const int                   N,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    const int                   incx,
                                    hipblasDoubleComplex*       y,
                                    const int                   incy);
#endif

// axpyBatched
hipblasStatus_t hipblasHaxpyBatchedFortran(hipblasHandle_t          handle,
                                           const int                N,
                                           const hipblasHalf*       alpha,
                                           const hipblasHalf* const x[],
                                           const int                incx,
                                           hipblasHalf* const       y[],
                                           const int                incy,
                                           const int                batch_count);

hipblasStatus_t hipblasSaxpyBatchedFortran(hipblasHandle_t    handle,
                                           const int          N,
                                           const float*       alpha,
                                           const float* const x[],
                                           const int          incx,
                                           float* const       y[],
                                           const int          incy,
                                           const int          batch_count);

hipblasStatus_t hipblasDaxpyBatchedFortran(hipblasHandle_t     handle,
                                           const int           N,
                                           const double*       alpha,
                                           const double* const x[],
                                           const int           incx,
                                           double* const       y[],
                                           const int           incy,
                                           const int           batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCaxpyBatchedFortran(hipblasHandle_t         handle,
                                           const int               N,
                                           const hipComplex*       alpha,
                                           const hipComplex* const x[],
                                           const int               incx,
                                           hipComplex* const       y[],
                                           const int               incy,
                                           const int               batch_count);

hipblasStatus_t hipblasZaxpyBatchedFortran(hipblasHandle_t               handle,
                                           const int                     N,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const x[],
                                           const int                     incx,
                                           hipDoubleComplex* const       y[],
                                           const int                     incy,
                                           const int                     batch_count);
#else
hipblasStatus_t hipblasCaxpyBatchedFortran(hipblasHandle_t             handle,
                                           const int                   N,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           const int                   incx,
                                           hipblasComplex* const       y[],
                                           const int                   incy,
                                           const int                   batch_count);

hipblasStatus_t hipblasZaxpyBatchedFortran(hipblasHandle_t                   handle,
                                           const int                         N,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           const int                         incx,
                                           hipblasDoubleComplex* const       y[],
                                           const int                         incy,
                                           const int                         batch_count);
#endif

// axpyStridedBatched
hipblasStatus_t hipblasHaxpyStridedBatchedFortran(hipblasHandle_t     handle,
                                                  const int           N,
                                                  const hipblasHalf*  alpha,
                                                  const hipblasHalf*  x,
                                                  const int           incx,
                                                  const hipblasStride stride_x,
                                                  hipblasHalf*        y,
                                                  const int           incy,
                                                  const hipblasStride stride_y,
                                                  const int           batch_count);

hipblasStatus_t hipblasSaxpyStridedBatchedFortran(hipblasHandle_t     handle,
                                                  const int           N,
                                                  const float*        alpha,
                                                  const float*        x,
                                                  const int           incx,
                                                  const hipblasStride stride_x,
                                                  float*              y,
                                                  const int           incy,
                                                  const hipblasStride stride_y,
                                                  const int           batch_count);

hipblasStatus_t hipblasDaxpyStridedBatchedFortran(hipblasHandle_t     handle,
                                                  const int           N,
                                                  const double*       alpha,
                                                  const double*       x,
                                                  const int           incx,
                                                  const hipblasStride stride_x,
                                                  double*             y,
                                                  const int           incy,
                                                  const hipblasStride stride_y,
                                                  const int           batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCaxpyStridedBatchedFortran(hipblasHandle_t     handle,
                                                  const int           N,
                                                  const hipComplex*   alpha,
                                                  const hipComplex*   x,
                                                  const int           incx,
                                                  const hipblasStride stride_x,
                                                  hipComplex*         y,
                                                  const int           incy,
                                                  const hipblasStride stride_y,
                                                  const int           batch_count);

hipblasStatus_t hipblasZaxpyStridedBatchedFortran(hipblasHandle_t         handle,
                                                  const int               N,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* x,
                                                  const int               incx,
                                                  const hipblasStride     stride_x,
                                                  hipDoubleComplex*       y,
                                                  const int               incy,
                                                  const hipblasStride     stride_y,
                                                  const int               batch_count);
#else
hipblasStatus_t hipblasCaxpyStridedBatchedFortran(hipblasHandle_t       handle,
                                                  const int             N,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  const int             incx,
                                                  const hipblasStride   stride_x,
                                                  hipblasComplex*       y,
                                                  const int             incy,
                                                  const hipblasStride   stride_y,
                                                  const int             batch_count);

hipblasStatus_t hipblasZaxpyStridedBatchedFortran(hipblasHandle_t             handle,
                                                  const int                   N,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  const int                   incx,
                                                  const hipblasStride         stride_x,
                                                  hipblasDoubleComplex*       y,
                                                  const int                   incy,
                                                  const hipblasStride         stride_y,
                                                  const int                   batch_count);
#endif

// asum
hipblasStatus_t
    hipblasSasumFortran(hipblasHandle_t handle, int n, const float* x, int incx, float* result);

hipblasStatus_t
    hipblasDasumFortran(hipblasHandle_t handle, int n, const double* x, int incx, double* result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasScasumFortran(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result);

hipblasStatus_t hipblasDzasumFortran(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result);
#else
hipblasStatus_t hipblasScasumFortran(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

hipblasStatus_t hipblasDzasumFortran(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);
#endif

// asumBatched
hipblasStatus_t hipblasSasumBatchedFortran(hipblasHandle_t    handle,
                                           int                n,
                                           const float* const x[],
                                           int                incx,
                                           int                batch_count,
                                           float*             results);

hipblasStatus_t hipblasDasumBatchedFortran(hipblasHandle_t     handle,
                                           int                 n,
                                           const double* const x[],
                                           int                 incx,
                                           int                 batch_count,
                                           double*             results);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasScasumBatchedFortran(hipblasHandle_t         handle,
                                            int                     n,
                                            const hipComplex* const x[],
                                            int                     incx,
                                            int                     batch_count,
                                            float*                  results);

hipblasStatus_t hipblasDzasumBatchedFortran(hipblasHandle_t               handle,
                                            int                           n,
                                            const hipDoubleComplex* const x[],
                                            int                           incx,
                                            int                           batch_count,
                                            double*                       results);
#else
hipblasStatus_t hipblasScasumBatchedFortran(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasComplex* const x[],
                                            int                         incx,
                                            int                         batch_count,
                                            float*                      results);

hipblasStatus_t hipblasDzasumBatchedFortran(hipblasHandle_t                   handle,
                                            int                               n,
                                            const hipblasDoubleComplex* const x[],
                                            int                               incx,
                                            int                               batch_count,
                                            double*                           results);
#endif

// asumStridedBatched
hipblasStatus_t hipblasSasumStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const float*    x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  float*          results);

hipblasStatus_t hipblasDasumStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  double*         results);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasScasumStridedBatchedFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const hipComplex* x,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   int               batch_count,
                                                   float*            results);

hipblasStatus_t hipblasDzasumStridedBatchedFortran(hipblasHandle_t         handle,
                                                   int                     n,
                                                   const hipDoubleComplex* x,
                                                   int                     incx,
                                                   hipblasStride           stridex,
                                                   int                     batch_count,
                                                   double*                 results);
#else
hipblasStatus_t hipblasScasumStridedBatchedFortran(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const hipblasComplex* x,
                                                   int                   incx,
                                                   hipblasStride         stridex,
                                                   int                   batch_count,
                                                   float*                results);

hipblasStatus_t hipblasDzasumStridedBatchedFortran(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasDoubleComplex* x,
                                                   int                         incx,
                                                   hipblasStride               stridex,
                                                   int                         batch_count,
                                                   double*                     results);
#endif

// nrm2
hipblasStatus_t
    hipblasSnrm2Fortran(hipblasHandle_t handle, int n, const float* x, int incx, float* result);

hipblasStatus_t
    hipblasDnrm2Fortran(hipblasHandle_t handle, int n, const double* x, int incx, double* result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasScnrm2Fortran(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result);

hipblasStatus_t hipblasDznrm2Fortran(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result);
#else
hipblasStatus_t hipblasScnrm2Fortran(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

hipblasStatus_t hipblasDznrm2Fortran(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);
#endif

// nrm2Batched
hipblasStatus_t hipblasSnrm2BatchedFortran(hipblasHandle_t    handle,
                                           int                n,
                                           const float* const x[],
                                           int                incx,
                                           int                batch_count,
                                           float*             results);

hipblasStatus_t hipblasDnrm2BatchedFortran(hipblasHandle_t     handle,
                                           int                 n,
                                           const double* const x[],
                                           int                 incx,
                                           int                 batch_count,
                                           double*             results);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasScnrm2BatchedFortran(hipblasHandle_t         handle,
                                            int                     n,
                                            const hipComplex* const x[],
                                            int                     incx,
                                            int                     batch_count,
                                            float*                  results);

hipblasStatus_t hipblasDznrm2BatchedFortran(hipblasHandle_t               handle,
                                            int                           n,
                                            const hipDoubleComplex* const x[],
                                            int                           incx,
                                            int                           batch_count,
                                            double*                       results);
#else
hipblasStatus_t hipblasScnrm2BatchedFortran(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasComplex* const x[],
                                            int                         incx,
                                            int                         batch_count,
                                            float*                      results);

hipblasStatus_t hipblasDznrm2BatchedFortran(hipblasHandle_t                   handle,
                                            int                               n,
                                            const hipblasDoubleComplex* const x[],
                                            int                               incx,
                                            int                               batch_count,
                                            double*                           results);
#endif

// nrm2StridedBatched
hipblasStatus_t hipblasSnrm2StridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const float*    x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  float*          results);

hipblasStatus_t hipblasDnrm2StridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  double*         results);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasScnrm2StridedBatchedFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const hipComplex* x,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   int               batch_count,
                                                   float*            results);

hipblasStatus_t hipblasDznrm2StridedBatchedFortran(hipblasHandle_t         handle,
                                                   int                     n,
                                                   const hipDoubleComplex* x,
                                                   int                     incx,
                                                   hipblasStride           stridex,
                                                   int                     batch_count,
                                                   double*                 results);
#else
hipblasStatus_t hipblasScnrm2StridedBatchedFortran(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const hipblasComplex* x,
                                                   int                   incx,
                                                   hipblasStride         stridex,
                                                   int                   batch_count,
                                                   float*                results);

hipblasStatus_t hipblasDznrm2StridedBatchedFortran(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasDoubleComplex* x,
                                                   int                         incx,
                                                   hipblasStride               stridex,
                                                   int                         batch_count,
                                                   double*                     results);
#endif

// amax
hipblasStatus_t
    hipblasIsamaxFortran(hipblasHandle_t handle, int n, const float* x, int incx, int* result);

hipblasStatus_t
    hipblasIdamaxFortran(hipblasHandle_t handle, int n, const double* x, int incx, int* result);

#ifdef HIPBLAS_V2
hipblasStatus_t
    hipblasIcamaxFortran(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzamaxFortran(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result);
#else
hipblasStatus_t hipblasIcamaxFortran(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzamaxFortran(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);
#endif

// amaxBatched
hipblasStatus_t hipblasIsamaxBatchedFortran(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result);

hipblasStatus_t hipblasIdamaxBatchedFortran(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasIcamaxBatchedFortran(hipblasHandle_t         handle,
                                            int                     n,
                                            const hipComplex* const x[],
                                            int                     incx,
                                            int                     batch_count,
                                            int*                    result);

hipblasStatus_t hipblasIzamaxBatchedFortran(hipblasHandle_t               handle,
                                            int                           n,
                                            const hipDoubleComplex* const x[],
                                            int                           incx,
                                            int                           batch_count,
                                            int*                          result);
#else
hipblasStatus_t hipblasIcamaxBatchedFortran(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasComplex* const x[],
                                            int                         incx,
                                            int                         batch_count,
                                            int*                        result);

hipblasStatus_t hipblasIzamaxBatchedFortran(hipblasHandle_t                   handle,
                                            int                               n,
                                            const hipblasDoubleComplex* const x[],
                                            int                               incx,
                                            int                               batch_count,
                                            int*                              result);
#endif

// amaxStridedBatched
hipblasStatus_t hipblasIsamaxStridedBatchedFortran(hipblasHandle_t handle,
                                                   int             n,
                                                   const float*    x,
                                                   int             incx,
                                                   hipblasStride   stridex,
                                                   int             batch_count,
                                                   int*            result);

hipblasStatus_t hipblasIdamaxStridedBatchedFortran(hipblasHandle_t handle,
                                                   int             n,
                                                   const double*   x,
                                                   int             incx,
                                                   hipblasStride   stridex,
                                                   int             batch_count,
                                                   int*            result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasIcamaxStridedBatchedFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const hipComplex* x,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   int               batch_count,
                                                   int*              result);

hipblasStatus_t hipblasIzamaxStridedBatchedFortran(hipblasHandle_t         handle,
                                                   int                     n,
                                                   const hipDoubleComplex* x,
                                                   int                     incx,
                                                   hipblasStride           stridex,
                                                   int                     batch_count,
                                                   int*                    result);
#else
hipblasStatus_t hipblasIcamaxStridedBatchedFortran(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const hipblasComplex* x,
                                                   int                   incx,
                                                   hipblasStride         stridex,
                                                   int                   batch_count,
                                                   int*                  result);

hipblasStatus_t hipblasIzamaxStridedBatchedFortran(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasDoubleComplex* x,
                                                   int                         incx,
                                                   hipblasStride               stridex,
                                                   int                         batch_count,
                                                   int*                        result);
#endif

// amin
hipblasStatus_t
    hipblasIsaminFortran(hipblasHandle_t handle, int n, const float* x, int incx, int* result);

hipblasStatus_t
    hipblasIdaminFortran(hipblasHandle_t handle, int n, const double* x, int incx, int* result);

#ifdef HIPBLAS_V2
hipblasStatus_t
    hipblasIcaminFortran(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzaminFortran(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result);
#else
hipblasStatus_t hipblasIcaminFortran(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzaminFortran(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);
#endif

// aminBatched
hipblasStatus_t hipblasIsaminBatchedFortran(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result);

hipblasStatus_t hipblasIdaminBatchedFortran(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasIcaminBatchedFortran(hipblasHandle_t         handle,
                                            int                     n,
                                            const hipComplex* const x[],
                                            int                     incx,
                                            int                     batch_count,
                                            int*                    result);
hipblasStatus_t hipblasIzaminBatchedFortran(hipblasHandle_t               handle,
                                            int                           n,
                                            const hipDoubleComplex* const x[],
                                            int                           incx,
                                            int                           batch_count,
                                            int*                          result);
#else
hipblasStatus_t hipblasIcaminBatchedFortran(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasComplex* const x[],
                                            int                         incx,
                                            int                         batch_count,
                                            int*                        result);
hipblasStatus_t hipblasIzaminBatchedFortran(hipblasHandle_t                   handle,
                                            int                               n,
                                            const hipblasDoubleComplex* const x[],
                                            int                               incx,
                                            int                               batch_count,
                                            int*                              result);
#endif

// aminStridedBatched
hipblasStatus_t hipblasIsaminStridedBatchedFortran(hipblasHandle_t handle,
                                                   int             n,
                                                   const float*    x,
                                                   int             incx,
                                                   hipblasStride   stridex,
                                                   int             batch_count,
                                                   int*            result);

hipblasStatus_t hipblasIdaminStridedBatchedFortran(hipblasHandle_t handle,
                                                   int             n,
                                                   const double*   x,
                                                   int             incx,
                                                   hipblasStride   stridex,
                                                   int             batch_count,
                                                   int*            result);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasIcaminStridedBatchedFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const hipComplex* x,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   int               batch_count,
                                                   int*              result);

hipblasStatus_t hipblasIzaminStridedBatchedFortran(hipblasHandle_t         handle,
                                                   int                     n,
                                                   const hipDoubleComplex* x,
                                                   int                     incx,
                                                   hipblasStride           stridex,
                                                   int                     batch_count,
                                                   int*                    result);
#else
hipblasStatus_t hipblasIcaminStridedBatchedFortran(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const hipblasComplex* x,
                                                   int                   incx,
                                                   hipblasStride         stridex,
                                                   int                   batch_count,
                                                   int*                  result);

hipblasStatus_t hipblasIzaminStridedBatchedFortran(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasDoubleComplex* x,
                                                   int                         incx,
                                                   hipblasStride               stridex,
                                                   int                         batch_count,
                                                   int*                        result);
#endif

// rot
hipblasStatus_t hipblasSrotFortran(hipblasHandle_t handle,
                                   int             n,
                                   float*          x,
                                   int             incx,
                                   float*          y,
                                   int             incy,
                                   const float*    c,
                                   const float*    s);

hipblasStatus_t hipblasDrotFortran(hipblasHandle_t handle,
                                   int             n,
                                   double*         x,
                                   int             incx,
                                   double*         y,
                                   int             incy,
                                   const double*   c,
                                   const double*   s);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCrotFortran(hipblasHandle_t   handle,
                                   int               n,
                                   hipComplex*       x,
                                   int               incx,
                                   hipComplex*       y,
                                   int               incy,
                                   const float*      c,
                                   const hipComplex* s);

hipblasStatus_t hipblasCsrotFortran(hipblasHandle_t handle,
                                    int             n,
                                    hipComplex*     x,
                                    int             incx,
                                    hipComplex*     y,
                                    int             incy,
                                    const float*    c,
                                    const float*    s);

hipblasStatus_t hipblasZrotFortran(hipblasHandle_t         handle,
                                   int                     n,
                                   hipDoubleComplex*       x,
                                   int                     incx,
                                   hipDoubleComplex*       y,
                                   int                     incy,
                                   const double*           c,
                                   const hipDoubleComplex* s);

hipblasStatus_t hipblasZdrotFortran(hipblasHandle_t   handle,
                                    int               n,
                                    hipDoubleComplex* x,
                                    int               incx,
                                    hipDoubleComplex* y,
                                    int               incy,
                                    const double*     c,
                                    const double*     s);
#else
hipblasStatus_t hipblasCrotFortran(hipblasHandle_t       handle,
                                   int                   n,
                                   hipblasComplex*       x,
                                   int                   incx,
                                   hipblasComplex*       y,
                                   int                   incy,
                                   const float*          c,
                                   const hipblasComplex* s);

hipblasStatus_t hipblasCsrotFortran(hipblasHandle_t handle,
                                    int             n,
                                    hipblasComplex* x,
                                    int             incx,
                                    hipblasComplex* y,
                                    int             incy,
                                    const float*    c,
                                    const float*    s);

hipblasStatus_t hipblasZrotFortran(hipblasHandle_t             handle,
                                   int                         n,
                                   hipblasDoubleComplex*       x,
                                   int                         incx,
                                   hipblasDoubleComplex*       y,
                                   int                         incy,
                                   const double*               c,
                                   const hipblasDoubleComplex* s);

hipblasStatus_t hipblasZdrotFortran(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasDoubleComplex* x,
                                    int                   incx,
                                    hipblasDoubleComplex* y,
                                    int                   incy,
                                    const double*         c,
                                    const double*         s);
#endif

// rotBatched
hipblasStatus_t hipblasSrotBatchedFortran(hipblasHandle_t handle,
                                          int             n,
                                          float* const    x[],
                                          int             incx,
                                          float* const    y[],
                                          int             incy,
                                          const float*    c,
                                          const float*    s,
                                          int             batch_count);

hipblasStatus_t hipblasDrotBatchedFortran(hipblasHandle_t handle,
                                          int             n,
                                          double* const   x[],
                                          int             incx,
                                          double* const   y[],
                                          int             incy,
                                          const double*   c,
                                          const double*   s,
                                          int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCrotBatchedFortran(hipblasHandle_t   handle,
                                          int               n,
                                          hipComplex* const x[],
                                          int               incx,
                                          hipComplex* const y[],
                                          int               incy,
                                          const float*      c,
                                          const hipComplex* s,
                                          int               batch_count);

hipblasStatus_t hipblasCsrotBatchedFortran(hipblasHandle_t   handle,
                                           int               n,
                                           hipComplex* const x[],
                                           int               incx,
                                           hipComplex* const y[],
                                           int               incy,
                                           const float*      c,
                                           const float*      s,
                                           int               batch_count);

hipblasStatus_t hipblasZrotBatchedFortran(hipblasHandle_t         handle,
                                          int                     n,
                                          hipDoubleComplex* const x[],
                                          int                     incx,
                                          hipDoubleComplex* const y[],
                                          int                     incy,
                                          const double*           c,
                                          const hipDoubleComplex* s,
                                          int                     batch_count);

hipblasStatus_t hipblasZdrotBatchedFortran(hipblasHandle_t         handle,
                                           int                     n,
                                           hipDoubleComplex* const x[],
                                           int                     incx,
                                           hipDoubleComplex* const y[],
                                           int                     incy,
                                           const double*           c,
                                           const double*           s,
                                           int                     batch_count);
#else
hipblasStatus_t hipblasCrotBatchedFortran(hipblasHandle_t       handle,
                                          int                   n,
                                          hipblasComplex* const x[],
                                          int                   incx,
                                          hipblasComplex* const y[],
                                          int                   incy,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int                   batch_count);

hipblasStatus_t hipblasCsrotBatchedFortran(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasComplex* const x[],
                                           int                   incx,
                                           hipblasComplex* const y[],
                                           int                   incy,
                                           const float*          c,
                                           const float*          s,
                                           int                   batch_count);

hipblasStatus_t hipblasZrotBatchedFortran(hipblasHandle_t             handle,
                                          int                         n,
                                          hipblasDoubleComplex* const x[],
                                          int                         incx,
                                          hipblasDoubleComplex* const y[],
                                          int                         incy,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int                         batch_count);

hipblasStatus_t hipblasZdrotBatchedFortran(hipblasHandle_t             handle,
                                           int                         n,
                                           hipblasDoubleComplex* const x[],
                                           int                         incx,
                                           hipblasDoubleComplex* const y[],
                                           int                         incy,
                                           const double*               c,
                                           const double*               s,
                                           int                         batch_count);
#endif

// rotStridedBatched
hipblasStatus_t hipblasSrotStridedBatchedFortran(hipblasHandle_t handle,
                                                 int             n,
                                                 float*          x,
                                                 int             incx,
                                                 hipblasStride   stride_x,
                                                 float*          y,
                                                 int             incy,
                                                 hipblasStride   stride_y,
                                                 const float*    c,
                                                 const float*    s,
                                                 int             batch_count);

hipblasStatus_t hipblasDrotStridedBatchedFortran(hipblasHandle_t handle,
                                                 int             n,
                                                 double*         x,
                                                 int             incx,
                                                 hipblasStride   stride_x,
                                                 double*         y,
                                                 int             incy,
                                                 hipblasStride   stride_y,
                                                 const double*   c,
                                                 const double*   s,
                                                 int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCrotStridedBatchedFortran(hipblasHandle_t   handle,
                                                 int               n,
                                                 hipComplex*       x,
                                                 int               incx,
                                                 hipblasStride     stride_x,
                                                 hipComplex*       y,
                                                 int               incy,
                                                 hipblasStride     stride_y,
                                                 const float*      c,
                                                 const hipComplex* s,
                                                 int               batch_count);

hipblasStatus_t hipblasCsrotStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  hipComplex*     x,
                                                  int             incx,
                                                  hipblasStride   stride_x,
                                                  hipComplex*     y,
                                                  int             incy,
                                                  hipblasStride   stride_y,
                                                  const float*    c,
                                                  const float*    s,
                                                  int             batch_count);

hipblasStatus_t hipblasZrotStridedBatchedFortran(hipblasHandle_t         handle,
                                                 int                     n,
                                                 hipDoubleComplex*       x,
                                                 int                     incx,
                                                 hipblasStride           stride_x,
                                                 hipDoubleComplex*       y,
                                                 int                     incy,
                                                 hipblasStride           stride_y,
                                                 const double*           c,
                                                 const hipDoubleComplex* s,
                                                 int                     batch_count);

hipblasStatus_t hipblasZdrotStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  hipDoubleComplex* x,
                                                  int               incx,
                                                  hipblasStride     stride_x,
                                                  hipDoubleComplex* y,
                                                  int               incy,
                                                  hipblasStride     stride_y,
                                                  const double*     c,
                                                  const double*     s,
                                                  int               batch_count);
#else
hipblasStatus_t hipblasCrotStridedBatchedFortran(hipblasHandle_t       handle,
                                                 int                   n,
                                                 hipblasComplex*       x,
                                                 int                   incx,
                                                 hipblasStride         stride_x,
                                                 hipblasComplex*       y,
                                                 int                   incy,
                                                 hipblasStride         stride_y,
                                                 const float*          c,
                                                 const hipblasComplex* s,
                                                 int                   batch_count);

hipblasStatus_t hipblasCsrotStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  hipblasComplex* x,
                                                  int             incx,
                                                  hipblasStride   stride_x,
                                                  hipblasComplex* y,
                                                  int             incy,
                                                  hipblasStride   stride_y,
                                                  const float*    c,
                                                  const float*    s,
                                                  int             batch_count);

hipblasStatus_t hipblasZrotStridedBatchedFortran(hipblasHandle_t             handle,
                                                 int                         n,
                                                 hipblasDoubleComplex*       x,
                                                 int                         incx,
                                                 hipblasStride               stride_x,
                                                 hipblasDoubleComplex*       y,
                                                 int                         incy,
                                                 hipblasStride               stride_y,
                                                 const double*               c,
                                                 const hipblasDoubleComplex* s,
                                                 int                         batch_count);

hipblasStatus_t hipblasZdrotStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   n,
                                                  hipblasDoubleComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  hipblasDoubleComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stride_y,
                                                  const double*         c,
                                                  const double*         s,
                                                  int                   batch_count);
#endif

// rotg
hipblasStatus_t hipblasSrotgFortran(hipblasHandle_t handle, float* a, float* b, float* c, float* s);

hipblasStatus_t
    hipblasDrotgFortran(hipblasHandle_t handle, double* a, double* b, double* c, double* s);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCrotgFortran(
    hipblasHandle_t handle, hipComplex* a, hipComplex* b, float* c, hipComplex* s);

hipblasStatus_t hipblasZrotgFortran(hipblasHandle_t   handle,
                                    hipDoubleComplex* a,
                                    hipDoubleComplex* b,
                                    double*           c,
                                    hipDoubleComplex* s);
#else
hipblasStatus_t hipblasCrotgFortran(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);

hipblasStatus_t hipblasZrotgFortran(hipblasHandle_t       handle,
                                    hipblasDoubleComplex* a,
                                    hipblasDoubleComplex* b,
                                    double*               c,
                                    hipblasDoubleComplex* s);
#endif

// rotgBatched
hipblasStatus_t hipblasSrotgBatchedFortran(hipblasHandle_t handle,
                                           float* const    a[],
                                           float* const    b[],
                                           float* const    c[],
                                           float* const    s[],
                                           int             batch_count);

hipblasStatus_t hipblasDrotgBatchedFortran(hipblasHandle_t handle,
                                           double* const   a[],
                                           double* const   b[],
                                           double* const   c[],
                                           double* const   s[],
                                           int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCrotgBatchedFortran(hipblasHandle_t   handle,
                                           hipComplex* const a[],
                                           hipComplex* const b[],
                                           float* const      c[],
                                           hipComplex* const s[],
                                           int               batch_count);

hipblasStatus_t hipblasZrotgBatchedFortran(hipblasHandle_t         handle,
                                           hipDoubleComplex* const a[],
                                           hipDoubleComplex* const b[],
                                           double* const           c[],
                                           hipDoubleComplex* const s[],
                                           int                     batch_count);
#else
hipblasStatus_t hipblasCrotgBatchedFortran(hipblasHandle_t       handle,
                                           hipblasComplex* const a[],
                                           hipblasComplex* const b[],
                                           float* const          c[],
                                           hipblasComplex* const s[],
                                           int                   batch_count);

hipblasStatus_t hipblasZrotgBatchedFortran(hipblasHandle_t             handle,
                                           hipblasDoubleComplex* const a[],
                                           hipblasDoubleComplex* const b[],
                                           double* const               c[],
                                           hipblasDoubleComplex* const s[],
                                           int                         batch_count);
#endif

// rotgStridedBatched
hipblasStatus_t hipblasSrotgStridedBatchedFortran(hipblasHandle_t handle,
                                                  float*          a,
                                                  hipblasStride   stride_a,
                                                  float*          b,
                                                  hipblasStride   stride_b,
                                                  float*          c,
                                                  hipblasStride   stride_c,
                                                  float*          s,
                                                  hipblasStride   stride_s,
                                                  int             batch_count);

hipblasStatus_t hipblasDrotgStridedBatchedFortran(hipblasHandle_t handle,
                                                  double*         a,
                                                  hipblasStride   stride_a,
                                                  double*         b,
                                                  hipblasStride   stride_b,
                                                  double*         c,
                                                  hipblasStride   stride_c,
                                                  double*         s,
                                                  hipblasStride   stride_s,
                                                  int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCrotgStridedBatchedFortran(hipblasHandle_t handle,
                                                  hipComplex*     a,
                                                  hipblasStride   stride_a,
                                                  hipComplex*     b,
                                                  hipblasStride   stride_b,
                                                  float*          c,
                                                  hipblasStride   stride_c,
                                                  hipComplex*     s,
                                                  hipblasStride   stride_s,
                                                  int             batch_count);

hipblasStatus_t hipblasZrotgStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipDoubleComplex* a,
                                                  hipblasStride     stride_a,
                                                  hipDoubleComplex* b,
                                                  hipblasStride     stride_b,
                                                  double*           c,
                                                  hipblasStride     stride_c,
                                                  hipDoubleComplex* s,
                                                  hipblasStride     stride_s,
                                                  int               batch_count);
#else
hipblasStatus_t hipblasCrotgStridedBatchedFortran(hipblasHandle_t handle,
                                                  hipblasComplex* a,
                                                  hipblasStride   stride_a,
                                                  hipblasComplex* b,
                                                  hipblasStride   stride_b,
                                                  float*          c,
                                                  hipblasStride   stride_c,
                                                  hipblasComplex* s,
                                                  hipblasStride   stride_s,
                                                  int             batch_count);

hipblasStatus_t hipblasZrotgStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasDoubleComplex* a,
                                                  hipblasStride         stride_a,
                                                  hipblasDoubleComplex* b,
                                                  hipblasStride         stride_b,
                                                  double*               c,
                                                  hipblasStride         stride_c,
                                                  hipblasDoubleComplex* s,
                                                  hipblasStride         stride_s,
                                                  int                   batch_count);
#endif

// rotm
hipblasStatus_t hipblasSrotmFortran(
    hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);

hipblasStatus_t hipblasDrotmFortran(
    hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);

// rotmBatched
hipblasStatus_t hipblasSrotmBatchedFortran(hipblasHandle_t    handle,
                                           int                n,
                                           float* const       x[],
                                           int                incx,
                                           float* const       y[],
                                           int                incy,
                                           const float* const param[],
                                           int                batch_count);

hipblasStatus_t hipblasDrotmBatchedFortran(hipblasHandle_t     handle,
                                           int                 n,
                                           double* const       x[],
                                           int                 incx,
                                           double* const       y[],
                                           int                 incy,
                                           const double* const param[],
                                           int                 batch_count);

// rotmStridedBatched
hipblasStatus_t hipblasSrotmStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  float*          x,
                                                  int             incx,
                                                  hipblasStride   stride_x,
                                                  float*          y,
                                                  int             incy,
                                                  hipblasStride   stride_y,
                                                  const float*    param,
                                                  hipblasStride   stride_param,
                                                  int             batch_count);

hipblasStatus_t hipblasDrotmStridedBatchedFortran(hipblasHandle_t handle,
                                                  int             n,
                                                  double*         x,
                                                  int             incx,
                                                  hipblasStride   stride_x,
                                                  double*         y,
                                                  int             incy,
                                                  hipblasStride   stride_y,
                                                  const double*   param,
                                                  hipblasStride   stride_param,
                                                  int             batch_count);

// rotmg
hipblasStatus_t hipblasSrotmgFortran(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);

hipblasStatus_t hipblasDrotmgFortran(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);

// rotmgBatched
hipblasStatus_t hipblasSrotmgBatchedFortran(hipblasHandle_t    handle,
                                            float* const       d1[],
                                            float* const       d2[],
                                            float* const       x1[],
                                            const float* const y1[],
                                            float* const       param[],
                                            int                batch_count);

hipblasStatus_t hipblasDrotmgBatchedFortran(hipblasHandle_t     handle,
                                            double* const       d1[],
                                            double* const       d2[],
                                            double* const       x1[],
                                            const double* const y1[],
                                            double* const       param[],
                                            int                 batch_count);

// rotmgStridedBatched
hipblasStatus_t hipblasSrotmgStridedBatchedFortran(hipblasHandle_t handle,
                                                   float*          d1,
                                                   hipblasStride   stride_d1,
                                                   float*          d2,
                                                   hipblasStride   stride_d2,
                                                   float*          x1,
                                                   hipblasStride   stride_x1,
                                                   const float*    y1,
                                                   hipblasStride   stride_y1,
                                                   float*          param,
                                                   hipblasStride   stride_param,
                                                   int             batch_count);

hipblasStatus_t hipblasDrotmgStridedBatchedFortran(hipblasHandle_t handle,
                                                   double*         d1,
                                                   hipblasStride   stride_d1,
                                                   double*         d2,
                                                   hipblasStride   stride_d2,
                                                   double*         x1,
                                                   hipblasStride   stride_x1,
                                                   const double*   y1,
                                                   hipblasStride   stride_y1,
                                                   double*         param,
                                                   hipblasStride   stride_param,
                                                   int             batch_count);

/* ==========
 *    L2
 * ========== */

// ger
hipblasStatus_t hipblasSgerFortran(hipblasHandle_t handle,
                                   int             m,
                                   int             n,
                                   const float*    alpha,
                                   const float*    x,
                                   int             incx,
                                   const float*    y,
                                   int             incy,
                                   float*          A,
                                   int             lda);

hipblasStatus_t hipblasDgerFortran(hipblasHandle_t handle,
                                   int             m,
                                   int             n,
                                   const double*   alpha,
                                   const double*   x,
                                   int             incx,
                                   const double*   y,
                                   int             incy,
                                   double*         A,
                                   int             lda);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeruFortran(hipblasHandle_t   handle,
                                    int               m,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       A,
                                    int               lda);

hipblasStatus_t hipblasCgercFortran(hipblasHandle_t   handle,
                                    int               m,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       A,
                                    int               lda);

hipblasStatus_t hipblasZgeruFortran(hipblasHandle_t         handle,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       A,
                                    int                     lda);

hipblasStatus_t hipblasZgercFortran(hipblasHandle_t         handle,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       A,
                                    int                     lda);
#else
hipblasStatus_t hipblasCgeruFortran(hipblasHandle_t       handle,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       A,
                                    int                   lda);

hipblasStatus_t hipblasCgercFortran(hipblasHandle_t       handle,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       A,
                                    int                   lda);

hipblasStatus_t hipblasZgeruFortran(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       A,
                                    int                         lda);

hipblasStatus_t hipblasZgercFortran(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       A,
                                    int                         lda);
#endif

// ger_batched
hipblasStatus_t hipblasSgerBatchedFortran(hipblasHandle_t    handle,
                                          int                m,
                                          int                n,
                                          const float*       alpha,
                                          const float* const x[],
                                          int                incx,
                                          const float* const y[],
                                          int                incy,
                                          float* const       A[],
                                          int                lda,
                                          int                batch_count);

hipblasStatus_t hipblasDgerBatchedFortran(hipblasHandle_t     handle,
                                          int                 m,
                                          int                 n,
                                          const double*       alpha,
                                          const double* const x[],
                                          int                 incx,
                                          const double* const y[],
                                          int                 incy,
                                          double* const       A[],
                                          int                 lda,
                                          int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeruBatchedFortran(hipblasHandle_t         handle,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           hipComplex* const       A[],
                                           int                     lda,
                                           int                     batch_count);

hipblasStatus_t hipblasCgercBatchedFortran(hipblasHandle_t         handle,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           hipComplex* const       A[],
                                           int                     lda,
                                           int                     batch_count);

hipblasStatus_t hipblasZgeruBatchedFortran(hipblasHandle_t               handle,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           hipDoubleComplex* const       A[],
                                           int                           lda,
                                           int                           batch_count);

hipblasStatus_t hipblasZgercBatchedFortran(hipblasHandle_t               handle,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           hipDoubleComplex* const       A[],
                                           int                           lda,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCgeruBatchedFortran(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           hipblasComplex* const       A[],
                                           int                         lda,
                                           int                         batch_count);

hipblasStatus_t hipblasCgercBatchedFortran(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           hipblasComplex* const       A[],
                                           int                         lda,
                                           int                         batch_count);

hipblasStatus_t hipblasZgeruBatchedFortran(hipblasHandle_t                   handle,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           hipblasDoubleComplex* const       A[],
                                           int                               lda,
                                           int                               batch_count);

hipblasStatus_t hipblasZgercBatchedFortran(hipblasHandle_t                   handle,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           hipblasDoubleComplex* const       A[],
                                           int                               lda,
                                           int                               batch_count);
#endif

// ger_strided_batched
hipblasStatus_t hipblasSgerStridedBatchedFortran(hipblasHandle_t handle,
                                                 int             m,
                                                 int             n,
                                                 const float*    alpha,
                                                 const float*    x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 const float*    y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 float*          A,
                                                 int             lda,
                                                 hipblasStride   strideA,
                                                 int             batch_count);

hipblasStatus_t hipblasDgerStridedBatchedFortran(hipblasHandle_t handle,
                                                 int             m,
                                                 int             n,
                                                 const double*   alpha,
                                                 const double*   x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 const double*   y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 double*         A,
                                                 int             lda,
                                                 hipblasStride   strideA,
                                                 int             batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeruStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               m,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  hipComplex*       A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  int               batch_count);

hipblasStatus_t hipblasCgercStridedBatchedFortran(hipblasHandle_t   handle,
                                                  int               m,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  hipComplex*       A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  int               batch_count);

hipblasStatus_t hipblasZgeruStridedBatchedFortran(hipblasHandle_t         handle,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  hipDoubleComplex*       A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  int                     batch_count);

hipblasStatus_t hipblasZgercStridedBatchedFortran(hipblasHandle_t         handle,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  hipDoubleComplex*       A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCgeruStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  int                   batch_count);

hipblasStatus_t hipblasCgercStridedBatchedFortran(hipblasHandle_t       handle,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  int                   batch_count);

hipblasStatus_t hipblasZgeruStridedBatchedFortran(hipblasHandle_t             handle,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  int                         batch_count);

hipblasStatus_t hipblasZgercStridedBatchedFortran(hipblasHandle_t             handle,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  int                         batch_count);
#endif

// hbmv
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChbmvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    int               k,
                                    const hipComplex* alpha,
                                    const hipComplex* A,
                                    int               lda,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* beta,
                                    hipComplex*       y,
                                    int               incy);

hipblasStatus_t hipblasZhbmvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    int                     k,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasChbmvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    int                   k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int                   incy);

hipblasStatus_t hipblasZhbmvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    int                         k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// hbmv_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChbmvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           int                     k,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex*       beta,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batchCount);

hipblasStatus_t hipblasZhbmvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           int                           k,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasChbmvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batchCount);

hipblasStatus_t hipblasZhbmvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           int                               k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batchCount);
#endif

// hbmv_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChbmvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  int               k,
                                                  const hipComplex* alpha,
                                                  const hipComplex* A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* beta,
                                                  hipComplex*       y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

hipblasStatus_t hipblasZhbmvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  int                     k,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasChbmvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  int                   k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batchCount);

hipblasStatus_t hipblasZhbmvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  int                         k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batchCount);
#endif

// hemv
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChemvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* A,
                                    int               lda,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* beta,
                                    hipComplex*       y,
                                    int               incy);

hipblasStatus_t hipblasZhemvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasChemvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int                   incy);

hipblasStatus_t hipblasZhemvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// hemv_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChemvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex*       beta,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batch_count);

hipblasStatus_t hipblasZhemvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasChemvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batch_count);

hipblasStatus_t hipblasZhemvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batch_count);
#endif

// hemv_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChemvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* A,
                                                  int               lda,
                                                  hipblasStride     stride_a,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stride_x,
                                                  const hipComplex* beta,
                                                  hipComplex*       y,
                                                  int               incy,
                                                  hipblasStride     stride_y,
                                                  int               batch_count);

hipblasStatus_t hipblasZhemvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           stride_a,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stride_x,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stride_y,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasChemvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         stride_a,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stride_y,
                                                  int                   batch_count);

hipblasStatus_t hipblasZhemvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               stride_a,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stride_x,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stride_y,
                                                  int                         batch_count);
#endif

// her
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const float*      alpha,
                                   const hipComplex* x,
                                   int               incx,
                                   hipComplex*       A,
                                   int               lda);

hipblasStatus_t hipblasZherFortran(hipblasHandle_t         handle,
                                   hipblasFillMode_t       uplo,
                                   int                     n,
                                   const double*           alpha,
                                   const hipDoubleComplex* x,
                                   int                     incx,
                                   hipDoubleComplex*       A,
                                   int                     lda);
#else
hipblasStatus_t hipblasCherFortran(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int                   n,
                                   const float*          alpha,
                                   const hipblasComplex* x,
                                   int                   incx,
                                   hipblasComplex*       A,
                                   int                   lda);

hipblasStatus_t hipblasZherFortran(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const double*               alpha,
                                   const hipblasDoubleComplex* x,
                                   int                         incx,
                                   hipblasDoubleComplex*       A,
                                   int                         lda);
#endif

// her_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherBatchedFortran(hipblasHandle_t         handle,
                                          hipblasFillMode_t       uplo,
                                          int                     n,
                                          const float*            alpha,
                                          const hipComplex* const x[],
                                          int                     incx,
                                          hipComplex* const       A[],
                                          int                     lda,
                                          int                     batchCount);

hipblasStatus_t hipblasZherBatchedFortran(hipblasHandle_t               handle,
                                          hipblasFillMode_t             uplo,
                                          int                           n,
                                          const double*                 alpha,
                                          const hipDoubleComplex* const x[],
                                          int                           incx,
                                          hipDoubleComplex* const       A[],
                                          int                           lda,
                                          int                           batchCount);
#else
hipblasStatus_t hipblasCherBatchedFortran(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const float*                alpha,
                                          const hipblasComplex* const x[],
                                          int                         incx,
                                          hipblasComplex* const       A[],
                                          int                         lda,
                                          int                         batchCount);

hipblasStatus_t hipblasZherBatchedFortran(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int                               n,
                                          const double*                     alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int                               incx,
                                          hipblasDoubleComplex* const       A[],
                                          int                               lda,
                                          int                               batchCount);
#endif

// her_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const float*      alpha,
                                                 const hipComplex* x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 hipComplex*       A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 int               batchCount);

hipblasStatus_t hipblasZherStridedBatchedFortran(hipblasHandle_t         handle,
                                                 hipblasFillMode_t       uplo,
                                                 int                     n,
                                                 const double*           alpha,
                                                 const hipDoubleComplex* x,
                                                 int                     incx,
                                                 hipblasStride           stridex,
                                                 hipDoubleComplex*       A,
                                                 int                     lda,
                                                 hipblasStride           strideA,
                                                 int                     batchCount);
#else
hipblasStatus_t hipblasCherStridedBatchedFortran(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int                   n,
                                                 const float*          alpha,
                                                 const hipblasComplex* x,
                                                 int                   incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       A,
                                                 int                   lda,
                                                 hipblasStride         strideA,
                                                 int                   batchCount);

hipblasStatus_t hipblasZherStridedBatchedFortran(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int                         n,
                                                 const double*               alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int                         incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       A,
                                                 int                         lda,
                                                 hipblasStride               strideA,
                                                 int                         batchCount);
#endif

// her2
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCher2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       A,
                                    int               lda);

hipblasStatus_t hipblasZher2Fortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       A,
                                    int                     lda);
#else
hipblasStatus_t hipblasCher2Fortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       A,
                                    int                   lda);

hipblasStatus_t hipblasZher2Fortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       A,
                                    int                         lda);
#endif

// her2_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCher2BatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           hipComplex* const       A[],
                                           int                     lda,
                                           int                     batchCount);

hipblasStatus_t hipblasZher2BatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           hipDoubleComplex* const       A[],
                                           int                           lda,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCher2BatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           hipblasComplex* const       A[],
                                           int                         lda,
                                           int                         batchCount);

hipblasStatus_t hipblasZher2BatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           hipblasDoubleComplex* const       A[],
                                           int                               lda,
                                           int                               batchCount);
#endif

// her2_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCher2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  hipComplex*       A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  int               batchCount);

hipblasStatus_t hipblasZher2StridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  hipDoubleComplex*       A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCher2StridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  int                   batchCount);

hipblasStatus_t hipblasZher2StridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  int                         batchCount);
#endif

// hpmv
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChpmvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* AP,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* beta,
                                    hipComplex*       y,
                                    int               incy);

hipblasStatus_t hipblasZhpmvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* AP,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasChpmvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* AP,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int                   incy);

hipblasStatus_t hipblasZhpmvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* AP,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// hpmv_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChpmvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const AP[],
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex*       beta,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batchCount);

hipblasStatus_t hipblasZhpmvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const AP[],
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasChpmvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const AP[],
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batchCount);

hipblasStatus_t hipblasZhpmvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const AP[],
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batchCount);
#endif

// hpmv_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChpmvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* AP,
                                                  hipblasStride     strideAP,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* beta,
                                                  hipComplex*       y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

hipblasStatus_t hipblasZhpmvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* AP,
                                                  hipblasStride           strideAP,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasChpmvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* AP,
                                                  hipblasStride         strideAP,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batchCount);

hipblasStatus_t hipblasZhpmvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* AP,
                                                  hipblasStride               strideAP,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batchCount);
#endif

// hpr
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChprFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const float*      alpha,
                                   const hipComplex* x,
                                   int               incx,
                                   hipComplex*       AP);

hipblasStatus_t hipblasZhprFortran(hipblasHandle_t         handle,
                                   hipblasFillMode_t       uplo,
                                   int                     n,
                                   const double*           alpha,
                                   const hipDoubleComplex* x,
                                   int                     incx,
                                   hipDoubleComplex*       AP);
#else
hipblasStatus_t hipblasChprFortran(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int                   n,
                                   const float*          alpha,
                                   const hipblasComplex* x,
                                   int                   incx,
                                   hipblasComplex*       AP);

hipblasStatus_t hipblasZhprFortran(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const double*               alpha,
                                   const hipblasDoubleComplex* x,
                                   int                         incx,
                                   hipblasDoubleComplex*       AP);
#endif

// hpr_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChprBatchedFortran(hipblasHandle_t         handle,
                                          hipblasFillMode_t       uplo,
                                          int                     n,
                                          const float*            alpha,
                                          const hipComplex* const x[],
                                          int                     incx,
                                          hipComplex* const       AP[],
                                          int                     batchCount);

hipblasStatus_t hipblasZhprBatchedFortran(hipblasHandle_t               handle,
                                          hipblasFillMode_t             uplo,
                                          int                           n,
                                          const double*                 alpha,
                                          const hipDoubleComplex* const x[],
                                          int                           incx,
                                          hipDoubleComplex* const       AP[],
                                          int                           batchCount);
#else
hipblasStatus_t hipblasChprBatchedFortran(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const float*                alpha,
                                          const hipblasComplex* const x[],
                                          int                         incx,
                                          hipblasComplex* const       AP[],
                                          int                         batchCount);

hipblasStatus_t hipblasZhprBatchedFortran(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int                               n,
                                          const double*                     alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int                               incx,
                                          hipblasDoubleComplex* const       AP[],
                                          int                               batchCount);
#endif

// hpr_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChprStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const float*      alpha,
                                                 const hipComplex* x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 hipComplex*       AP,
                                                 hipblasStride     strideAP,
                                                 int               batchCount);

hipblasStatus_t hipblasZhprStridedBatchedFortran(hipblasHandle_t         handle,
                                                 hipblasFillMode_t       uplo,
                                                 int                     n,
                                                 const double*           alpha,
                                                 const hipDoubleComplex* x,
                                                 int                     incx,
                                                 hipblasStride           stridex,
                                                 hipDoubleComplex*       AP,
                                                 hipblasStride           strideAP,
                                                 int                     batchCount);
#else
hipblasStatus_t hipblasChprStridedBatchedFortran(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int                   n,
                                                 const float*          alpha,
                                                 const hipblasComplex* x,
                                                 int                   incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       AP,
                                                 hipblasStride         strideAP,
                                                 int                   batchCount);

hipblasStatus_t hipblasZhprStridedBatchedFortran(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int                         n,
                                                 const double*               alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int                         incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       AP,
                                                 hipblasStride               strideAP,
                                                 int                         batchCount);
#endif

// hpr2
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChpr2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       AP);

hipblasStatus_t hipblasZhpr2Fortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       AP);
#else
hipblasStatus_t hipblasChpr2Fortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       AP);

hipblasStatus_t hipblasZhpr2Fortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       AP);
#endif

// hpr2_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChpr2BatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           hipComplex* const       AP[],
                                           int                     batchCount);

hipblasStatus_t hipblasZhpr2BatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           hipDoubleComplex* const       AP[],
                                           int                           batchCount);
#else
hipblasStatus_t hipblasChpr2BatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           hipblasComplex* const       AP[],
                                           int                         batchCount);

hipblasStatus_t hipblasZhpr2BatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           hipblasDoubleComplex* const       AP[],
                                           int                               batchCount);
#endif

// hpr2_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChpr2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  hipComplex*       AP,
                                                  hipblasStride     strideAP,
                                                  int               batchCount);

hipblasStatus_t hipblasZhpr2StridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  hipDoubleComplex*       AP,
                                                  hipblasStride           strideAP,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasChpr2StridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       AP,
                                                  hipblasStride         strideAP,
                                                  int                   batchCount);

hipblasStatus_t hipblasZhpr2StridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       AP,
                                                  hipblasStride               strideAP,
                                                  int                         batchCount);
#endif

// sbmv
hipblasStatus_t hipblasSsbmvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    int               k,
                                    const float*      alpha,
                                    const float*      A,
                                    int               lda,
                                    const float*      x,
                                    int               incx,
                                    const float*      beta,
                                    float*            y,
                                    int               incy);

hipblasStatus_t hipblasDsbmvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    int               k,
                                    const double*     alpha,
                                    const double*     A,
                                    int               lda,
                                    const double*     x,
                                    int               incx,
                                    const double*     beta,
                                    double*           y,
                                    int               incy);

// sbmv_batched
hipblasStatus_t hipblasSsbmvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const x[],
                                           int                incx,
                                           const float*       beta,
                                           float*             y[],
                                           int                incy,
                                           int                batchCount);

hipblasStatus_t hipblasDsbmvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           int                 n,
                                           int                 k,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const x[],
                                           int                 incx,
                                           const double*       beta,
                                           double*             y[],
                                           int                 incy,
                                           int                 batchCount);

// sbmv_strided_batched
hipblasStatus_t hipblasSsbmvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  int               k,
                                                  const float*      alpha,
                                                  const float*      A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const float*      x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const float*      beta,
                                                  float*            y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

hipblasStatus_t hipblasDsbmvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  int               k,
                                                  const double*     alpha,
                                                  const double*     A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const double*     x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const double*     beta,
                                                  double*           y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

// spmv
hipblasStatus_t hipblasSspmvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const float*      alpha,
                                    const float*      AP,
                                    const float*      x,
                                    int               incx,
                                    const float*      beta,
                                    float*            y,
                                    int               incy);

hipblasStatus_t hipblasDspmvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const double*     alpha,
                                    const double*     AP,
                                    const double*     x,
                                    int               incx,
                                    const double*     beta,
                                    double*           y,
                                    int               incy);

// spmv_batched
hipblasStatus_t hipblasSspmvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           int                n,
                                           const float*       alpha,
                                           const float* const AP[],
                                           const float* const x[],
                                           int                incx,
                                           const float*       beta,
                                           float*             y[],
                                           int                incy,
                                           int                batchCount);

hipblasStatus_t hipblasDspmvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const AP[],
                                           const double* const x[],
                                           int                 incx,
                                           const double*       beta,
                                           double*             y[],
                                           int                 incy,
                                           int                 batchCount);

// spmv_strided_batched
hipblasStatus_t hipblasSspmvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const float*      alpha,
                                                  const float*      AP,
                                                  hipblasStride     strideAP,
                                                  const float*      x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const float*      beta,
                                                  float*            y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

hipblasStatus_t hipblasDspmvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const double*     alpha,
                                                  const double*     AP,
                                                  hipblasStride     strideAP,
                                                  const double*     x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const double*     beta,
                                                  double*           y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

// spr
hipblasStatus_t hipblasSsprFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const float*      alpha,
                                   const float*      x,
                                   int               incx,
                                   float*            AP);

hipblasStatus_t hipblasDsprFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const double*     alpha,
                                   const double*     x,
                                   int               incx,
                                   double*           AP);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsprFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const hipComplex* alpha,
                                   const hipComplex* x,
                                   int               incx,
                                   hipComplex*       AP);

hipblasStatus_t hipblasZsprFortran(hipblasHandle_t         handle,
                                   hipblasFillMode_t       uplo,
                                   int                     n,
                                   const hipDoubleComplex* alpha,
                                   const hipDoubleComplex* x,
                                   int                     incx,
                                   hipDoubleComplex*       AP);
#else
hipblasStatus_t hipblasCsprFortran(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int                   n,
                                   const hipblasComplex* alpha,
                                   const hipblasComplex* x,
                                   int                   incx,
                                   hipblasComplex*       AP);

hipblasStatus_t hipblasZsprFortran(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasDoubleComplex* alpha,
                                   const hipblasDoubleComplex* x,
                                   int                         incx,
                                   hipblasDoubleComplex*       AP);
#endif

// spr_batched
hipblasStatus_t hipblasSsprBatchedFortran(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          int                n,
                                          const float*       alpha,
                                          const float* const x[],
                                          int                incx,
                                          float* const       AP[],
                                          int                batchCount);

hipblasStatus_t hipblasDsprBatchedFortran(hipblasHandle_t     handle,
                                          hipblasFillMode_t   uplo,
                                          int                 n,
                                          const double*       alpha,
                                          const double* const x[],
                                          int                 incx,
                                          double* const       AP[],
                                          int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsprBatchedFortran(hipblasHandle_t         handle,
                                          hipblasFillMode_t       uplo,
                                          int                     n,
                                          const hipComplex*       alpha,
                                          const hipComplex* const x[],
                                          int                     incx,
                                          hipComplex* const       AP[],
                                          int                     batchCount);

hipblasStatus_t hipblasZsprBatchedFortran(hipblasHandle_t               handle,
                                          hipblasFillMode_t             uplo,
                                          int                           n,
                                          const hipDoubleComplex*       alpha,
                                          const hipDoubleComplex* const x[],
                                          int                           incx,
                                          hipDoubleComplex* const       AP[],
                                          int                           batchCount);
#else
hipblasStatus_t hipblasCsprBatchedFortran(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasComplex*       alpha,
                                          const hipblasComplex* const x[],
                                          int                         incx,
                                          hipblasComplex* const       AP[],
                                          int                         batchCount);

hipblasStatus_t hipblasZsprBatchedFortran(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int                               n,
                                          const hipblasDoubleComplex*       alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int                               incx,
                                          hipblasDoubleComplex* const       AP[],
                                          int                               batchCount);
#endif

// spr_strided_batched
hipblasStatus_t hipblasSsprStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const float*      alpha,
                                                 const float*      x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 float*            AP,
                                                 hipblasStride     strideAP,
                                                 int               batchCount);

hipblasStatus_t hipblasDsprStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const double*     alpha,
                                                 const double*     x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 double*           AP,
                                                 hipblasStride     strideAP,
                                                 int               batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsprStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const hipComplex* alpha,
                                                 const hipComplex* x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 hipComplex*       AP,
                                                 hipblasStride     strideAP,
                                                 int               batchCount);

hipblasStatus_t hipblasZsprStridedBatchedFortran(hipblasHandle_t         handle,
                                                 hipblasFillMode_t       uplo,
                                                 int                     n,
                                                 const hipDoubleComplex* alpha,
                                                 const hipDoubleComplex* x,
                                                 int                     incx,
                                                 hipblasStride           stridex,
                                                 hipDoubleComplex*       AP,
                                                 hipblasStride           strideAP,
                                                 int                     batchCount);
#else
hipblasStatus_t hipblasCsprStridedBatchedFortran(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int                   n,
                                                 const hipblasComplex* alpha,
                                                 const hipblasComplex* x,
                                                 int                   incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       AP,
                                                 hipblasStride         strideAP,
                                                 int                   batchCount);

hipblasStatus_t hipblasZsprStridedBatchedFortran(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int                         n,
                                                 const hipblasDoubleComplex* alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int                         incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       AP,
                                                 hipblasStride               strideAP,
                                                 int                         batchCount);
#endif

// spr2
hipblasStatus_t hipblasSspr2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const float*      alpha,
                                    const float*      x,
                                    int               incx,
                                    const float*      y,
                                    int               incy,
                                    float*            AP);

hipblasStatus_t hipblasDspr2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const double*     alpha,
                                    const double*     x,
                                    int               incx,
                                    const double*     y,
                                    int               incy,
                                    double*           AP);

// spr2_batched
hipblasStatus_t hipblasSspr2BatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           int                n,
                                           const float*       alpha,
                                           const float* const x[],
                                           int                incx,
                                           const float* const y[],
                                           int                incy,
                                           float* const       AP[],
                                           int                batchCount);

hipblasStatus_t hipblasDspr2BatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const x[],
                                           int                 incx,
                                           const double* const y[],
                                           int                 incy,
                                           double* const       AP[],
                                           int                 batchCount);

// spr2_strided_batched
hipblasStatus_t hipblasSspr2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const float*      alpha,
                                                  const float*      x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const float*      y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  float*            AP,
                                                  hipblasStride     strideAP,
                                                  int               batchCount);

hipblasStatus_t hipblasDspr2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const double*     alpha,
                                                  const double*     x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const double*     y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  double*           AP,
                                                  hipblasStride     strideAP,
                                                  int               batchCount);

// symv
hipblasStatus_t hipblasSsymvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const float*      alpha,
                                    const float*      A,
                                    int               lda,
                                    const float*      x,
                                    int               incx,
                                    const float*      beta,
                                    float*            y,
                                    int               incy);

hipblasStatus_t hipblasDsymvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const double*     alpha,
                                    const double*     A,
                                    int               lda,
                                    const double*     x,
                                    int               incx,
                                    const double*     beta,
                                    double*           y,
                                    int               incy);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsymvFortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* A,
                                    int               lda,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* beta,
                                    hipComplex*       y,
                                    int               incy);

hipblasStatus_t hipblasZsymvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasCsymvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int                   incy);

hipblasStatus_t hipblasZsymvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// symv_batched
hipblasStatus_t hipblasSsymvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           int                n,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const x[],
                                           int                incx,
                                           const float*       beta,
                                           float* const       y[],
                                           int                incy,
                                           int                batchCount);

hipblasStatus_t hipblasDsymvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const x[],
                                           int                 incx,
                                           const double*       beta,
                                           double* const       y[],
                                           int                 incy,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsymvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex*       beta,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batchCount);

hipblasStatus_t hipblasZsymvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCsymvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batchCount);

hipblasStatus_t hipblasZsymvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batchCount);
#endif

// symv_strided_batched
hipblasStatus_t hipblasSsymvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const float*      alpha,
                                                  const float*      A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const float*      x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const float*      beta,
                                                  float*            y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

hipblasStatus_t hipblasDsymvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const double*     alpha,
                                                  const double*     A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const double*     x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const double*     beta,
                                                  double*           y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsymvStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* beta,
                                                  hipComplex*       y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batchCount);

hipblasStatus_t hipblasZsymvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCsymvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batchCount);

hipblasStatus_t hipblasZsymvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batchCount);
#endif

// syr
hipblasStatus_t hipblasSsyrFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const float*      alpha,
                                   const float*      x,
                                   int               incx,
                                   float*            A,
                                   int               lda);

hipblasStatus_t hipblasDsyrFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const double*     alpha,
                                   const double*     x,
                                   int               incx,
                                   double*           A,
                                   int               lda);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrFortran(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const hipComplex* alpha,
                                   const hipComplex* x,
                                   int               incx,
                                   hipComplex*       A,
                                   int               lda);

hipblasStatus_t hipblasZsyrFortran(hipblasHandle_t         handle,
                                   hipblasFillMode_t       uplo,
                                   int                     n,
                                   const hipDoubleComplex* alpha,
                                   const hipDoubleComplex* x,
                                   int                     incx,
                                   hipDoubleComplex*       A,
                                   int                     lda);
#else
hipblasStatus_t hipblasCsyrFortran(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int                   n,
                                   const hipblasComplex* alpha,
                                   const hipblasComplex* x,
                                   int                   incx,
                                   hipblasComplex*       A,
                                   int                   lda);

hipblasStatus_t hipblasZsyrFortran(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasDoubleComplex* alpha,
                                   const hipblasDoubleComplex* x,
                                   int                         incx,
                                   hipblasDoubleComplex*       A,
                                   int                         lda);
#endif

// syr_batched
hipblasStatus_t hipblasSsyrBatchedFortran(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          int                n,
                                          const float*       alpha,
                                          const float* const x[],
                                          int                incx,
                                          float* const       A[],
                                          int                lda,
                                          int                batch_count);

hipblasStatus_t hipblasDsyrBatchedFortran(hipblasHandle_t     handle,
                                          hipblasFillMode_t   uplo,
                                          int                 n,
                                          const double*       alpha,
                                          const double* const x[],
                                          int                 incx,
                                          double* const       A[],
                                          int                 lda,
                                          int                 batch_count);
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrBatchedFortran(hipblasHandle_t         handle,
                                          hipblasFillMode_t       uplo,
                                          int                     n,
                                          const hipComplex*       alpha,
                                          const hipComplex* const x[],
                                          int                     incx,
                                          hipComplex* const       A[],
                                          int                     lda,
                                          int                     batch_count);

hipblasStatus_t hipblasZsyrBatchedFortran(hipblasHandle_t               handle,
                                          hipblasFillMode_t             uplo,
                                          int                           n,
                                          const hipDoubleComplex*       alpha,
                                          const hipDoubleComplex* const x[],
                                          int                           incx,
                                          hipDoubleComplex* const       A[],
                                          int                           lda,
                                          int                           batch_count);
#else
hipblasStatus_t hipblasCsyrBatchedFortran(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasComplex*       alpha,
                                          const hipblasComplex* const x[],
                                          int                         incx,
                                          hipblasComplex* const       A[],
                                          int                         lda,
                                          int                         batch_count);

hipblasStatus_t hipblasZsyrBatchedFortran(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int                               n,
                                          const hipblasDoubleComplex*       alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int                               incx,
                                          hipblasDoubleComplex* const       A[],
                                          int                               lda,
                                          int                               batch_count);
#endif

// syr_strided_batched
hipblasStatus_t hipblasSsyrStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const float*      alpha,
                                                 const float*      x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 float*            A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 int               batch_count);

hipblasStatus_t hipblasDsyrStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const double*     alpha,
                                                 const double*     x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 double*           A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 int               batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrStridedBatchedFortran(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const hipComplex* alpha,
                                                 const hipComplex* x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 hipComplex*       A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 int               batch_count);

hipblasStatus_t hipblasZsyrStridedBatchedFortran(hipblasHandle_t         handle,
                                                 hipblasFillMode_t       uplo,
                                                 int                     n,
                                                 const hipDoubleComplex* alpha,
                                                 const hipDoubleComplex* x,
                                                 int                     incx,
                                                 hipblasStride           stridex,
                                                 hipDoubleComplex*       A,
                                                 int                     lda,
                                                 hipblasStride           strideA,
                                                 int                     batch_count);
#else
hipblasStatus_t hipblasCsyrStridedBatchedFortran(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int                   n,
                                                 const hipblasComplex* alpha,
                                                 const hipblasComplex* x,
                                                 int                   incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       A,
                                                 int                   lda,
                                                 hipblasStride         strideA,
                                                 int                   batch_count);

hipblasStatus_t hipblasZsyrStridedBatchedFortran(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int                         n,
                                                 const hipblasDoubleComplex* alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int                         incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       A,
                                                 int                         lda,
                                                 hipblasStride               strideA,
                                                 int                         batch_count);
#endif

// syr2
hipblasStatus_t hipblasSsyr2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const float*      alpha,
                                    const float*      x,
                                    int               incx,
                                    const float*      y,
                                    int               incy,
                                    float*            A,
                                    int               lda);

hipblasStatus_t hipblasDsyr2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const double*     alpha,
                                    const double*     x,
                                    int               incx,
                                    const double*     y,
                                    int               incy,
                                    double*           A,
                                    int               lda);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyr2Fortran(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* x,
                                    int               incx,
                                    const hipComplex* y,
                                    int               incy,
                                    hipComplex*       A,
                                    int               lda);

hipblasStatus_t hipblasZsyr2Fortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* y,
                                    int                     incy,
                                    hipDoubleComplex*       A,
                                    int                     lda);
#else
hipblasStatus_t hipblasCsyr2Fortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* y,
                                    int                   incy,
                                    hipblasComplex*       A,
                                    int                   lda);

hipblasStatus_t hipblasZsyr2Fortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* y,
                                    int                         incy,
                                    hipblasDoubleComplex*       A,
                                    int                         lda);
#endif

// syr2_batched
hipblasStatus_t hipblasSsyr2BatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           int                n,
                                           const float*       alpha,
                                           const float* const x[],
                                           int                incx,
                                           const float* const y[],
                                           int                incy,
                                           float* const       A[],
                                           int                lda,
                                           int                batchCount);

hipblasStatus_t hipblasDsyr2BatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const x[],
                                           int                 incx,
                                           const double* const y[],
                                           int                 incy,
                                           double* const       A[],
                                           int                 lda,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyr2BatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex* const y[],
                                           int                     incy,
                                           hipComplex* const       A[],
                                           int                     lda,
                                           int                     batchCount);

hipblasStatus_t hipblasZsyr2BatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex* const y[],
                                           int                           incy,
                                           hipDoubleComplex* const       A[],
                                           int                           lda,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCsyr2BatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex* const y[],
                                           int                         incy,
                                           hipblasComplex* const       A[],
                                           int                         lda,
                                           int                         batchCount);

hipblasStatus_t hipblasZsyr2BatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex* const y[],
                                           int                               incy,
                                           hipblasDoubleComplex* const       A[],
                                           int                               lda,
                                           int                               batchCount);
#endif

// syr2_strided_batched
hipblasStatus_t hipblasSsyr2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const float*      alpha,
                                                  const float*      x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const float*      y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  float*            A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  int               batchCount);

hipblasStatus_t hipblasDsyr2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const double*     alpha,
                                                  const double*     x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const double*     y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  double*           A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  int               batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyr2StridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const hipComplex* y,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  hipComplex*       A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  int               batchCount);

hipblasStatus_t hipblasZsyr2StridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  hipDoubleComplex*       A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCsyr2StridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  int                   batchCount);

hipblasStatus_t hipblasZsyr2StridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  int                         batchCount);
#endif

// tbmv
hipblasStatus_t hipblasStbmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const float*       A,
                                    int                lda,
                                    float*             x,
                                    int                incx);

hipblasStatus_t hipblasDtbmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtbmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const hipComplex*  A,
                                    int                lda,
                                    hipComplex*        x,
                                    int                incx);

hipblasStatus_t hipblasZtbmvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    int                     k,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    hipDoubleComplex*       x,
                                    int                     incx);
#else
hipblasStatus_t hipblasCtbmvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    int                   k,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    hipblasComplex*       x,
                                    int                   incx);

hipblasStatus_t hipblasZtbmvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         k,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);
#endif

// tbmv_batched
hipblasStatus_t hipblasStbmvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const float* const A[],
                                           int                lda,
                                           float* const       x[],
                                           int                incx,
                                           int                batch_count);

hipblasStatus_t hipblasDtbmvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           int                 k,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       x[],
                                           int                 incx,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtbmvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           int                     k,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           hipComplex* const       x[],
                                           int                     incx,
                                           int                     batch_count);

hipblasStatus_t hipblasZtbmvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           int                           k,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           hipDoubleComplex* const       x[],
                                           int                           incx,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCtbmvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         k,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           hipblasComplex* const       x[],
                                           int                         incx,
                                           int                         batch_count);

hipblasStatus_t hipblasZtbmvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           int                               k,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           hipblasDoubleComplex* const       x[],
                                           int                               incx,
                                           int                               batch_count);
#endif

// tbmv_strided_batched
hipblasStatus_t hipblasStbmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  float*             x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  int                batch_count);

hipblasStatus_t hipblasDtbmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  double*            x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtbmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  hipComplex*        x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  int                batch_count);

hipblasStatus_t hipblasZtbmvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  int                     k,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           stride_a,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stride_x,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCtbmvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  int                   k,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         stride_a,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  int                   batch_count);

hipblasStatus_t hipblasZtbmvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  int                         k,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               stride_a,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stride_x,
                                                  int                         batch_count);
#endif

// tbsv
hipblasStatus_t hipblasStbsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const float*       A,
                                    int                lda,
                                    float*             x,
                                    int                incx);

hipblasStatus_t hipblasDtbsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtbsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const hipComplex*  A,
                                    int                lda,
                                    hipComplex*        x,
                                    int                incx);

hipblasStatus_t hipblasZtbsvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    int                     k,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    hipDoubleComplex*       x,
                                    int                     incx);
#else
hipblasStatus_t hipblasCtbsvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    int                   k,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    hipblasComplex*       x,
                                    int                   incx);

hipblasStatus_t hipblasZtbsvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         k,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);
#endif

// tbsv_batched
hipblasStatus_t hipblasStbsvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const float* const A[],
                                           int                lda,
                                           float* const       x[],
                                           int                incx,
                                           int                batchCount);

hipblasStatus_t hipblasDtbsvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           int                 k,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       x[],
                                           int                 incx,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtbsvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           int                     k,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           hipComplex* const       x[],
                                           int                     incx,
                                           int                     batchCount);

hipblasStatus_t hipblasZtbsvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           int                           k,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           hipDoubleComplex* const       x[],
                                           int                           incx,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCtbsvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         k,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           hipblasComplex* const       x[],
                                           int                         incx,
                                           int                         batchCount);

hipblasStatus_t hipblasZtbsvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           int                               k,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           hipblasDoubleComplex* const       x[],
                                           int                               incx,
                                           int                               batchCount);
#endif

// tbsv_strided_batched
hipblasStatus_t hipblasStbsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  float*             x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

hipblasStatus_t hipblasDtbsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  double*            x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtbsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  hipComplex*        x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

hipblasStatus_t hipblasZtbsvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  int                     k,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCtbsvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  int                   k,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  int                   batchCount);

hipblasStatus_t hipblasZtbsvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  int                         k,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  int                         batchCount);
#endif

// tpmv
hipblasStatus_t hipblasStpmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float*       AP,
                                    float*             x,
                                    int                incx);

hipblasStatus_t hipblasDtpmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const double*      AP,
                                    double*            x,
                                    int                incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtpmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const hipComplex*  AP,
                                    hipComplex*        x,
                                    int                incx);

hipblasStatus_t hipblasZtpmvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    const hipDoubleComplex* AP,
                                    hipDoubleComplex*       x,
                                    int                     incx);
#else
hipblasStatus_t hipblasCtpmvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    const hipblasComplex* AP,
                                    hipblasComplex*       x,
                                    int                   incx);

hipblasStatus_t hipblasZtpmvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasDoubleComplex* AP,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);
#endif

// tpmv_batched
hipblasStatus_t hipblasStpmvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float* const AP[],
                                           float* const       x[],
                                           int                incx,
                                           int                batchCount);

hipblasStatus_t hipblasDtpmvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           const double* const AP[],
                                           double* const       x[],
                                           int                 incx,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtpmvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           const hipComplex* const AP[],
                                           hipComplex* const       x[],
                                           int                     incx,
                                           int                     batchCount);

hipblasStatus_t hipblasZtpmvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           const hipDoubleComplex* const AP[],
                                           hipDoubleComplex* const       x[],
                                           int                           incx,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCtpmvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasComplex* const AP[],
                                           hipblasComplex* const       x[],
                                           int                         incx,
                                           int                         batchCount);

hipblasStatus_t hipblasZtpmvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           const hipblasDoubleComplex* const AP[],
                                           hipblasDoubleComplex* const       x[],
                                           int                               incx,
                                           int                               batchCount);
#endif

// tpmv_strided_batched
hipblasStatus_t hipblasStpmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const float*       AP,
                                                  hipblasStride      strideAP,
                                                  float*             x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

hipblasStatus_t hipblasDtpmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const double*      AP,
                                                  hipblasStride      strideAP,
                                                  double*            x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtpmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const hipComplex*  AP,
                                                  hipblasStride      strideAP,
                                                  hipComplex*        x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

hipblasStatus_t hipblasZtpmvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  const hipDoubleComplex* AP,
                                                  hipblasStride           strideAP,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCtpmvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  const hipblasComplex* AP,
                                                  hipblasStride         strideAP,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  int                   batchCount);

hipblasStatus_t hipblasZtpmvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  const hipblasDoubleComplex* AP,
                                                  hipblasStride               strideAP,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  int                         batchCount);
#endif

// tpsv
hipblasStatus_t hipblasStpsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float*       AP,
                                    float*             x,
                                    int                incx);

hipblasStatus_t hipblasDtpsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const double*      AP,
                                    double*            x,
                                    int                incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtpsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const hipComplex*  AP,
                                    hipComplex*        x,
                                    int                incx);

hipblasStatus_t hipblasZtpsvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    const hipDoubleComplex* AP,
                                    hipDoubleComplex*       x,
                                    int                     incx);
#else
hipblasStatus_t hipblasCtpsvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    const hipblasComplex* AP,
                                    hipblasComplex*       x,
                                    int                   incx);

hipblasStatus_t hipblasZtpsvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasDoubleComplex* AP,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);
#endif

// tpsv_bathced
hipblasStatus_t hipblasStpsvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float* const AP[],
                                           float* const       x[],
                                           int                incx,
                                           int                batchCount);

hipblasStatus_t hipblasDtpsvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           const double* const AP[],
                                           double* const       x[],
                                           int                 incx,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtpsvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           const hipComplex* const AP[],
                                           hipComplex* const       x[],
                                           int                     incx,
                                           int                     batchCount);

hipblasStatus_t hipblasZtpsvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           const hipDoubleComplex* const AP[],
                                           hipDoubleComplex* const       x[],
                                           int                           incx,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCtpsvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasComplex* const AP[],
                                           hipblasComplex* const       x[],
                                           int                         incx,
                                           int                         batchCount);

hipblasStatus_t hipblasZtpsvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           const hipblasDoubleComplex* const AP[],
                                           hipblasDoubleComplex* const       x[],
                                           int                               incx,
                                           int                               batchCount);
#endif

// tpsv_strided_batched
hipblasStatus_t hipblasStpsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const float*       AP,
                                                  hipblasStride      strideAP,
                                                  float*             x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

hipblasStatus_t hipblasDtpsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const double*      AP,
                                                  hipblasStride      strideAP,
                                                  double*            x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtpsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const hipComplex*  AP,
                                                  hipblasStride      strideAP,
                                                  hipComplex*        x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batchCount);

hipblasStatus_t hipblasZtpsvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  const hipDoubleComplex* AP,
                                                  hipblasStride           strideAP,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCtpsvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  const hipblasComplex* AP,
                                                  hipblasStride         strideAP,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  int                   batchCount);

hipblasStatus_t hipblasZtpsvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  const hipblasDoubleComplex* AP,
                                                  hipblasStride               strideAP,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  int                         batchCount);
#endif

// trmv
hipblasStatus_t hipblasStrmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float*       A,
                                    int                lda,
                                    float*             x,
                                    int                incx);

hipblasStatus_t hipblasDtrmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrmvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const hipComplex*  A,
                                    int                lda,
                                    hipComplex*        x,
                                    int                incx);

hipblasStatus_t hipblasZtrmvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    hipDoubleComplex*       x,
                                    int                     incx);
#else
hipblasStatus_t hipblasCtrmvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    hipblasComplex*       x,
                                    int                   incx);

hipblasStatus_t hipblasZtrmvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);
#endif

// trmv_batched
hipblasStatus_t hipblasStrmvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float* const A[],
                                           int                lda,
                                           float* const       x[],
                                           int                incx,
                                           int                batch_count);

hipblasStatus_t hipblasDtrmvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       x[],
                                           int                 incx,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrmvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           hipComplex* const       x[],
                                           int                     incx,
                                           int                     batch_count);

hipblasStatus_t hipblasZtrmvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           hipDoubleComplex* const       x[],
                                           int                           incx,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCtrmvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           hipblasComplex* const       x[],
                                           int                         incx,
                                           int                         batch_count);

hipblasStatus_t hipblasZtrmvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           hipblasDoubleComplex* const       x[],
                                           int                               incx,
                                           int                               batch_count);
#endif

// trmv_strided_batched
hipblasStatus_t hipblasStrmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  float*             x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  int                batch_count);

hipblasStatus_t hipblasDtrmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  double*            x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  hipComplex*        x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  int                batch_count);

hipblasStatus_t hipblasZtrmvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           stride_a,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stride_x,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCtrmvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         stride_a,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  int                   batch_count);

hipblasStatus_t hipblasZtrmvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               stride_a,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stride_x,
                                                  int                         batch_count);
#endif

// trsv
hipblasStatus_t hipblasStrsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const float*       A,
                                    int                lda,
                                    float*             x,
                                    int                incx);

hipblasStatus_t hipblasDtrsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrsvFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const hipComplex*  A,
                                    int                lda,
                                    hipComplex*        x,
                                    int                incx);

hipblasStatus_t hipblasZtrsvFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    hipDoubleComplex*       x,
                                    int                     incx);
#else
hipblasStatus_t hipblasCtrsvFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    hipblasComplex*       x,
                                    int                   incx);

hipblasStatus_t hipblasZtrsvFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    hipblasDoubleComplex*       x,
                                    int                         incx);
#endif

// trsv_batched
hipblasStatus_t hipblasStrsvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float* const A[],
                                           int                lda,
                                           float* const       x[],
                                           int                incx,
                                           int                batch_count);

hipblasStatus_t hipblasDtrsvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       x[],
                                           int                 incx,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrsvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           hipComplex* const       x[],
                                           int                     incx,
                                           int                     batch_count);

hipblasStatus_t hipblasZtrsvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           hipDoubleComplex* const       x[],
                                           int                           incx,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCtrsvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           hipblasComplex* const       x[],
                                           int                         incx,
                                           int                         batch_count);

hipblasStatus_t hipblasZtrsvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           hipblasDoubleComplex* const       x[],
                                           int                               incx,
                                           int                               batch_count);
#endif

// trsv_strided_batched
hipblasStatus_t hipblasStrsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  float*             x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batch_count);

hipblasStatus_t hipblasDtrsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  double*            x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrsvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  hipComplex*        x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  int                batch_count);

hipblasStatus_t hipblasZtrsvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  hipDoubleComplex*       x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCtrsvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  int                   batch_count);

hipblasStatus_t hipblasZtrsvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  int                         batch_count);
#endif

// gbmv
hipblasStatus_t hipblasSgbmvFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    int                kl,
                                    int                ku,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    const float*       x,
                                    int                incx,
                                    const float*       beta,
                                    float*             y,
                                    int                incy);

hipblasStatus_t hipblasDgbmvFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    int                kl,
                                    int                ku,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      x,
                                    int                incx,
                                    const double*      beta,
                                    double*            y,
                                    int                incy);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgbmvFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    int                kl,
                                    int                ku,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const hipComplex*  x,
                                    int                incx,
                                    const hipComplex*  beta,
                                    hipComplex*        y,
                                    int                incy);

hipblasStatus_t hipblasZgbmvFortran(hipblasHandle_t         handle,
                                    hipblasOperation_t      transA,
                                    int                     m,
                                    int                     n,
                                    int                     kl,
                                    int                     ku,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasCgbmvFortran(hipblasHandle_t       handle,
                                    hipblasOperation_t    transA,
                                    int                   m,
                                    int                   n,
                                    int                   kl,
                                    int                   ku,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int                   incy);

hipblasStatus_t hipblasZgbmvFortran(hipblasHandle_t             handle,
                                    hipblasOperation_t          transA,
                                    int                         m,
                                    int                         n,
                                    int                         kl,
                                    int                         ku,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// gbmv_batched
hipblasStatus_t hipblasSgbmvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t transA,
                                           int                m,
                                           int                n,
                                           int                kl,
                                           int                ku,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const x[],
                                           int                incx,
                                           const float*       beta,
                                           float* const       y[],
                                           int                incy,
                                           int                batch_count);

hipblasStatus_t hipblasDgbmvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasOperation_t  transA,
                                           int                 m,
                                           int                 n,
                                           int                 kl,
                                           int                 ku,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const x[],
                                           int                 incx,
                                           const double*       beta,
                                           double* const       y[],
                                           int                 incy,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgbmvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasOperation_t      transA,
                                           int                     m,
                                           int                     n,
                                           int                     kl,
                                           int                     ku,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex*       beta,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batch_count);

hipblasStatus_t hipblasZgbmvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasOperation_t            transA,
                                           int                           m,
                                           int                           n,
                                           int                           kl,
                                           int                           ku,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCgbmvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasOperation_t          transA,
                                           int                         m,
                                           int                         n,
                                           int                         kl,
                                           int                         ku,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batch_count);

hipblasStatus_t hipblasZgbmvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transA,
                                           int                               m,
                                           int                               n,
                                           int                               kl,
                                           int                               ku,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batch_count);
#endif

// gbmv_strided_batched
hipblasStatus_t hipblasSgbmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  int                kl,
                                                  int                ku,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  const float*       x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  const float*       beta,
                                                  float*             y,
                                                  int                incy,
                                                  hipblasStride      stride_y,
                                                  int                batch_count);

hipblasStatus_t hipblasDgbmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  int                kl,
                                                  int                ku,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  const double*      x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  const double*      beta,
                                                  double*            y,
                                                  int                incy,
                                                  hipblasStride      stride_y,
                                                  int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgbmvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  int                kl,
                                                  int                ku,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      stride_a,
                                                  const hipComplex*  x,
                                                  int                incx,
                                                  hipblasStride      stride_x,
                                                  const hipComplex*  beta,
                                                  hipComplex*        y,
                                                  int                incy,
                                                  hipblasStride      stride_y,
                                                  int                batch_count);

hipblasStatus_t hipblasZgbmvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasOperation_t      transA,
                                                  int                     m,
                                                  int                     n,
                                                  int                     kl,
                                                  int                     ku,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           stride_a,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stride_x,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stride_y,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCgbmvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transA,
                                                  int                   m,
                                                  int                   n,
                                                  int                   kl,
                                                  int                   ku,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         stride_a,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stride_y,
                                                  int                   batch_count);

hipblasStatus_t hipblasZgbmvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  int                         m,
                                                  int                         n,
                                                  int                         kl,
                                                  int                         ku,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               stride_a,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stride_x,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stride_y,
                                                  int                         batch_count);
#endif

// gemv
hipblasStatus_t hipblasSgemvFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    const float*       x,
                                    int                incx,
                                    const float*       beta,
                                    float*             y,
                                    int                incy);

hipblasStatus_t hipblasDgemvFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      x,
                                    int                incx,
                                    const double*      beta,
                                    double*            y,
                                    int                incy);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgemvFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const hipComplex*  x,
                                    int                incx,
                                    const hipComplex*  beta,
                                    hipComplex*        y,
                                    int                incy);

hipblasStatus_t hipblasZgemvFortran(hipblasHandle_t         handle,
                                    hipblasOperation_t      transA,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       y,
                                    int                     incy);
#else
hipblasStatus_t hipblasCgemvFortran(hipblasHandle_t       handle,
                                    hipblasOperation_t    transA,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int                   incy);

hipblasStatus_t hipblasZgemvFortran(hipblasHandle_t             handle,
                                    hipblasOperation_t          transA,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int                         incy);
#endif

// gemv_batched
hipblasStatus_t hipblasSgemvBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t transA,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const x[],
                                           int                incx,
                                           const float*       beta,
                                           float* const       y[],
                                           int                incy,
                                           int                batch_count);

hipblasStatus_t hipblasDgemvBatchedFortran(hipblasHandle_t     handle,
                                           hipblasOperation_t  transA,
                                           int                 m,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const x[],
                                           int                 incx,
                                           const double*       beta,
                                           double* const       y[],
                                           int                 incy,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgemvBatchedFortran(hipblasHandle_t         handle,
                                           hipblasOperation_t      transA,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           const hipComplex*       beta,
                                           hipComplex* const       y[],
                                           int                     incy,
                                           int                     batch_count);

hipblasStatus_t hipblasZgemvBatchedFortran(hipblasHandle_t               handle,
                                           hipblasOperation_t            transA,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       y[],
                                           int                           incy,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCgemvBatchedFortran(hipblasHandle_t             handle,
                                           hipblasOperation_t          transA,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int                         incy,
                                           int                         batch_count);

hipblasStatus_t hipblasZgemvBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transA,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int                               incy,
                                           int                               batch_count);
#endif

// gemv_strided_batched
hipblasStatus_t hipblasSgemvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const float*       x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  const float*       beta,
                                                  float*             y,
                                                  int                incy,
                                                  hipblasStride      stridey,
                                                  int                batch_count);

hipblasStatus_t hipblasDgemvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const double*      x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  const double*      beta,
                                                  double*            y,
                                                  int                incy,
                                                  hipblasStride      stridey,
                                                  int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgemvStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const hipComplex*  x,
                                                  int                incx,
                                                  hipblasStride      stridex,
                                                  const hipComplex*  beta,
                                                  hipComplex*        y,
                                                  int                incy,
                                                  hipblasStride      stridey,
                                                  int                batch_count);

hipblasStatus_t hipblasZgemvStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasOperation_t      transA,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stridex,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       y,
                                                  int                     incy,
                                                  hipblasStride           stridey,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCgemvStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transA,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  hipblasStride         stridey,
                                                  int                   batch_count);

hipblasStatus_t hipblasZgemvStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy,
                                                  hipblasStride               stridey,
                                                  int                         batch_count);
#endif

/* ==========
 *    L3
 * ========== */

// herk
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherkFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const float*       beta,
                                    hipComplex*        C,
                                    int                ldc);

hipblasStatus_t hipblasZherkFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    int                     n,
                                    int                     k,
                                    const double*           alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const double*           beta,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCherkFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    int                   n,
                                    int                   k,
                                    const float*          alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const float*          beta,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZherkFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int                         n,
                                    int                         k,
                                    const double*               alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const double*               beta,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// herk_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherkBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           int                     n,
                                           int                     k,
                                           const float*            alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const float*            beta,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZherkBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           int                           n,
                                           int                           k,
                                           const double*                 alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const double*                 beta,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCherkBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const float*                alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const float*                beta,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZherkBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           int                               n,
                                           int                               k,
                                           const double*                     alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const double*                     beta,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// herk_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherkStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const float*       alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const float*       beta,
                                                  hipComplex*        C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasZherkStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  int                     n,
                                                  int                     k,
                                                  const double*           alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const double*           beta,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           strideC,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCherkStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  int                   n,
                                                  int                   k,
                                                  const float*          alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const float*          beta,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         strideC,
                                                  int                   batchCount);

hipblasStatus_t hipblasZherkStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  int                         n,
                                                  int                         k,
                                                  const double*               alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const double*               beta,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               strideC,
                                                  int                         batchCount);
#endif

// herkx
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherkxFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const hipComplex*  alpha,
                                     const hipComplex*  A,
                                     int                lda,
                                     const hipComplex*  B,
                                     int                ldb,
                                     const float*       beta,
                                     hipComplex*        C,
                                     int                ldc);

hipblasStatus_t hipblasZherkxFortran(hipblasHandle_t         handle,
                                     hipblasFillMode_t       uplo,
                                     hipblasOperation_t      transA,
                                     int                     n,
                                     int                     k,
                                     const hipDoubleComplex* alpha,
                                     const hipDoubleComplex* A,
                                     int                     lda,
                                     const hipDoubleComplex* B,
                                     int                     ldb,
                                     const double*           beta,
                                     hipDoubleComplex*       C,
                                     int                     ldc);
#else
hipblasStatus_t hipblasCherkxFortran(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int                   n,
                                     int                   k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int                   lda,
                                     const hipblasComplex* B,
                                     int                   ldb,
                                     const float*          beta,
                                     hipblasComplex*       C,
                                     int                   ldc);

hipblasStatus_t hipblasZherkxFortran(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int                         lda,
                                     const hipblasDoubleComplex* B,
                                     int                         ldb,
                                     const double*               beta,
                                     hipblasDoubleComplex*       C,
                                     int                         ldc);
#endif

// herkx_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherkxBatchedFortran(hipblasHandle_t         handle,
                                            hipblasFillMode_t       uplo,
                                            hipblasOperation_t      transA,
                                            int                     n,
                                            int                     k,
                                            const hipComplex*       alpha,
                                            const hipComplex* const A[],
                                            int                     lda,
                                            const hipComplex* const B[],
                                            int                     ldb,
                                            const float*            beta,
                                            hipComplex* const       C[],
                                            int                     ldc,
                                            int                     batchCount);

hipblasStatus_t hipblasZherkxBatchedFortran(hipblasHandle_t               handle,
                                            hipblasFillMode_t             uplo,
                                            hipblasOperation_t            transA,
                                            int                           n,
                                            int                           k,
                                            const hipDoubleComplex*       alpha,
                                            const hipDoubleComplex* const A[],
                                            int                           lda,
                                            const hipDoubleComplex* const B[],
                                            int                           ldb,
                                            const double*                 beta,
                                            hipDoubleComplex* const       C[],
                                            int                           ldc,
                                            int                           batchCount);
#else
hipblasStatus_t hipblasCherkxBatchedFortran(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int                         lda,
                                            const hipblasComplex* const B[],
                                            int                         ldb,
                                            const float*                beta,
                                            hipblasComplex* const       C[],
                                            int                         ldc,
                                            int                         batchCount);

hipblasStatus_t hipblasZherkxBatchedFortran(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int                               n,
                                            int                               k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int                               lda,
                                            const hipblasDoubleComplex* const B[],
                                            int                               ldb,
                                            const double*                     beta,
                                            hipblasDoubleComplex* const       C[],
                                            int                               ldc,
                                            int                               batchCount);
#endif

// herkx_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCherkxStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const hipComplex*  alpha,
                                                   const hipComplex*  A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const hipComplex*  B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const float*       beta,
                                                   hipComplex*        C,
                                                   int                ldc,
                                                   hipblasStride      strideC,
                                                   int                batchCount);

hipblasStatus_t hipblasZherkxStridedBatchedFortran(hipblasHandle_t         handle,
                                                   hipblasFillMode_t       uplo,
                                                   hipblasOperation_t      transA,
                                                   int                     n,
                                                   int                     k,
                                                   const hipDoubleComplex* alpha,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   hipblasStride           strideA,
                                                   const hipDoubleComplex* B,
                                                   int                     ldb,
                                                   hipblasStride           strideB,
                                                   const double*           beta,
                                                   hipDoubleComplex*       C,
                                                   int                     ldc,
                                                   hipblasStride           strideC,
                                                   int                     batchCount);
#else
hipblasStatus_t hipblasCherkxStridedBatchedFortran(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int                   n,
                                                   int                   k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int                   lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int                   ldb,
                                                   hipblasStride         strideB,
                                                   const float*          beta,
                                                   hipblasComplex*       C,
                                                   int                   ldc,
                                                   hipblasStride         strideC,
                                                   int                   batchCount);

hipblasStatus_t hipblasZherkxStridedBatchedFortran(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int                         lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int                         ldb,
                                                   hipblasStride               strideB,
                                                   const double*               beta,
                                                   hipblasDoubleComplex*       C,
                                                   int                         ldc,
                                                   hipblasStride               strideC,
                                                   int                         batchCount);
#endif

// her2k
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCher2kFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const hipComplex*  alpha,
                                     const hipComplex*  A,
                                     int                lda,
                                     const hipComplex*  B,
                                     int                ldb,
                                     const float*       beta,
                                     hipComplex*        C,
                                     int                ldc);

hipblasStatus_t hipblasZher2kFortran(hipblasHandle_t         handle,
                                     hipblasFillMode_t       uplo,
                                     hipblasOperation_t      transA,
                                     int                     n,
                                     int                     k,
                                     const hipDoubleComplex* alpha,
                                     const hipDoubleComplex* A,
                                     int                     lda,
                                     const hipDoubleComplex* B,
                                     int                     ldb,
                                     const double*           beta,
                                     hipDoubleComplex*       C,
                                     int                     ldc);
#else
hipblasStatus_t hipblasCher2kFortran(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int                   n,
                                     int                   k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int                   lda,
                                     const hipblasComplex* B,
                                     int                   ldb,
                                     const float*          beta,
                                     hipblasComplex*       C,
                                     int                   ldc);

hipblasStatus_t hipblasZher2kFortran(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int                         lda,
                                     const hipblasDoubleComplex* B,
                                     int                         ldb,
                                     const double*               beta,
                                     hipblasDoubleComplex*       C,
                                     int                         ldc);
#endif

// her2k_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCher2kBatchedFortran(hipblasHandle_t         handle,
                                            hipblasFillMode_t       uplo,
                                            hipblasOperation_t      transA,
                                            int                     n,
                                            int                     k,
                                            const hipComplex*       alpha,
                                            const hipComplex* const A[],
                                            int                     lda,
                                            const hipComplex* const B[],
                                            int                     ldb,
                                            const float*            beta,
                                            hipComplex* const       C[],
                                            int                     ldc,
                                            int                     batchCount);

hipblasStatus_t hipblasZher2kBatchedFortran(hipblasHandle_t               handle,
                                            hipblasFillMode_t             uplo,
                                            hipblasOperation_t            transA,
                                            int                           n,
                                            int                           k,
                                            const hipDoubleComplex*       alpha,
                                            const hipDoubleComplex* const A[],
                                            int                           lda,
                                            const hipDoubleComplex* const B[],
                                            int                           ldb,
                                            const double*                 beta,
                                            hipDoubleComplex* const       C[],
                                            int                           ldc,
                                            int                           batchCount);
#else
hipblasStatus_t hipblasCher2kBatchedFortran(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int                         lda,
                                            const hipblasComplex* const B[],
                                            int                         ldb,
                                            const float*                beta,
                                            hipblasComplex* const       C[],
                                            int                         ldc,
                                            int                         batchCount);

hipblasStatus_t hipblasZher2kBatchedFortran(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int                               n,
                                            int                               k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int                               lda,
                                            const hipblasDoubleComplex* const B[],
                                            int                               ldb,
                                            const double*                     beta,
                                            hipblasDoubleComplex* const       C[],
                                            int                               ldc,
                                            int                               batchCount);
#endif

// her2k_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCher2kStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const hipComplex*  alpha,
                                                   const hipComplex*  A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const hipComplex*  B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const float*       beta,
                                                   hipComplex*        C,
                                                   int                ldc,
                                                   hipblasStride      strideC,
                                                   int                batchCount);

hipblasStatus_t hipblasZher2kStridedBatchedFortran(hipblasHandle_t         handle,
                                                   hipblasFillMode_t       uplo,
                                                   hipblasOperation_t      transA,
                                                   int                     n,
                                                   int                     k,
                                                   const hipDoubleComplex* alpha,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   hipblasStride           strideA,
                                                   const hipDoubleComplex* B,
                                                   int                     ldb,
                                                   hipblasStride           strideB,
                                                   const double*           beta,
                                                   hipDoubleComplex*       C,
                                                   int                     ldc,
                                                   hipblasStride           strideC,
                                                   int                     batchCount);
#else
hipblasStatus_t hipblasCher2kStridedBatchedFortran(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int                   n,
                                                   int                   k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int                   lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int                   ldb,
                                                   hipblasStride         strideB,
                                                   const float*          beta,
                                                   hipblasComplex*       C,
                                                   int                   ldc,
                                                   hipblasStride         strideC,
                                                   int                   batchCount);

hipblasStatus_t hipblasZher2kStridedBatchedFortran(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int                         lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int                         ldb,
                                                   hipblasStride               strideB,
                                                   const double*               beta,
                                                   hipblasDoubleComplex*       C,
                                                   int                         ldc,
                                                   hipblasStride               strideC,
                                                   int                         batchCount);
#endif

// symm
hipblasStatus_t hipblasSsymmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    hipblasFillMode_t uplo,
                                    int               m,
                                    int               n,
                                    const float*      alpha,
                                    const float*      A,
                                    int               lda,
                                    const float*      B,
                                    int               ldb,
                                    const float*      beta,
                                    float*            C,
                                    int               ldc);

hipblasStatus_t hipblasDsymmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    hipblasFillMode_t uplo,
                                    int               m,
                                    int               n,
                                    const double*     alpha,
                                    const double*     A,
                                    int               lda,
                                    const double*     B,
                                    int               ldb,
                                    const double*     beta,
                                    double*           C,
                                    int               ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsymmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    hipblasFillMode_t uplo,
                                    int               m,
                                    int               n,
                                    const hipComplex* alpha,
                                    const hipComplex* A,
                                    int               lda,
                                    const hipComplex* B,
                                    int               ldb,
                                    const hipComplex* beta,
                                    hipComplex*       C,
                                    int               ldc);

hipblasStatus_t hipblasZsymmFortran(hipblasHandle_t         handle,
                                    hipblasSideMode_t       side,
                                    hipblasFillMode_t       uplo,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* B,
                                    int                     ldb,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCsymmFortran(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* B,
                                    int                   ldb,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZsymmFortran(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* B,
                                    int                         ldb,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// symm_batched
hipblasStatus_t hipblasSsymmBatchedFortran(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const B[],
                                           int                ldb,
                                           const float*       beta,
                                           float* const       C[],
                                           int                ldc,
                                           int                batchCount);

hipblasStatus_t hipblasDsymmBatchedFortran(hipblasHandle_t     handle,
                                           hipblasSideMode_t   side,
                                           hipblasFillMode_t   uplo,
                                           int                 m,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const B[],
                                           int                 ldb,
                                           const double*       beta,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsymmBatchedFortran(hipblasHandle_t         handle,
                                           hipblasSideMode_t       side,
                                           hipblasFillMode_t       uplo,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const B[],
                                           int                     ldb,
                                           const hipComplex*       beta,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZsymmBatchedFortran(hipblasHandle_t               handle,
                                           hipblasSideMode_t             side,
                                           hipblasFillMode_t             uplo,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const B[],
                                           int                           ldb,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCsymmBatchedFortran(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const B[],
                                           int                         ldb,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZsymmBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const B[],
                                           int                               ldb,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// symm_strided_batched
hipblasStatus_t hipblasSsymmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  hipblasFillMode_t uplo,
                                                  int               m,
                                                  int               n,
                                                  const float*      alpha,
                                                  const float*      A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const float*      B,
                                                  int               ldb,
                                                  hipblasStride     strideB,
                                                  const float*      beta,
                                                  float*            C,
                                                  int               ldc,
                                                  hipblasStride     strideC,
                                                  int               batchCount);

hipblasStatus_t hipblasDsymmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  hipblasFillMode_t uplo,
                                                  int               m,
                                                  int               n,
                                                  const double*     alpha,
                                                  const double*     A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const double*     B,
                                                  int               ldb,
                                                  hipblasStride     strideB,
                                                  const double*     beta,
                                                  double*           C,
                                                  int               ldc,
                                                  hipblasStride     strideC,
                                                  int               batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsymmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  hipblasFillMode_t uplo,
                                                  int               m,
                                                  int               n,
                                                  const hipComplex* alpha,
                                                  const hipComplex* A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const hipComplex* B,
                                                  int               ldb,
                                                  hipblasStride     strideB,
                                                  const hipComplex* beta,
                                                  hipComplex*       C,
                                                  int               ldc,
                                                  hipblasStride     strideC,
                                                  int               batchCount);

hipblasStatus_t hipblasZsymmStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasSideMode_t       side,
                                                  hipblasFillMode_t       uplo,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* B,
                                                  int                     ldb,
                                                  hipblasStride           strideB,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           strideC,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCsymmStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* B,
                                                  int                   ldb,
                                                  hipblasStride         strideB,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         strideC,
                                                  int                   batchCount);

hipblasStatus_t hipblasZsymmStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* B,
                                                  int                         ldb,
                                                  hipblasStride               strideB,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               strideC,
                                                  int                         batchCount);
#endif

// syrk
hipblasStatus_t hipblasSsyrkFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    const float*       beta,
                                    float*             C,
                                    int                ldc);

hipblasStatus_t hipblasDsyrkFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      beta,
                                    double*            C,
                                    int                ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrkFortran(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const hipComplex*  beta,
                                    hipComplex*        C,
                                    int                ldc);

hipblasStatus_t hipblasZsyrkFortran(hipblasHandle_t         handle,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    int                     n,
                                    int                     k,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCsyrkFortran(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    int                   n,
                                    int                   k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZsyrkFortran(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int                         n,
                                    int                         k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// syrk_batched
hipblasStatus_t hipblasSsyrkBatchedFortran(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float*       beta,
                                           float* const       C[],
                                           int                ldc,
                                           int                batchCount);

hipblasStatus_t hipblasDsyrkBatchedFortran(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           int                 n,
                                           int                 k,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double*       beta,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrkBatchedFortran(hipblasHandle_t         handle,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           int                     n,
                                           int                     k,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex*       beta,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZsyrkBatchedFortran(hipblasHandle_t               handle,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           int                           n,
                                           int                           k,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCsyrkBatchedFortran(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZsyrkBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           int                               n,
                                           int                               k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// syrk_strided_batched
hipblasStatus_t hipblasSsyrkStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const float*       beta,
                                                  float*             C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasDsyrkStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const double*      beta,
                                                  double*            C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrkStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const hipComplex*  beta,
                                                  hipComplex*        C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasZsyrkStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  int                     n,
                                                  int                     k,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           strideC,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCsyrkStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  int                   n,
                                                  int                   k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         strideC,
                                                  int                   batchCount);

hipblasStatus_t hipblasZsyrkStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  int                         n,
                                                  int                         k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               strideC,
                                                  int                         batchCount);
#endif

// syr2k
hipblasStatus_t hipblasSsyr2kFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const float*       alpha,
                                     const float*       A,
                                     int                lda,
                                     const float*       B,
                                     int                ldb,
                                     const float*       beta,
                                     float*             C,
                                     int                ldc);

hipblasStatus_t hipblasDsyr2kFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const double*      alpha,
                                     const double*      A,
                                     int                lda,
                                     const double*      B,
                                     int                ldb,
                                     const double*      beta,
                                     double*            C,
                                     int                ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyr2kFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const hipComplex*  alpha,
                                     const hipComplex*  A,
                                     int                lda,
                                     const hipComplex*  B,
                                     int                ldb,
                                     const hipComplex*  beta,
                                     hipComplex*        C,
                                     int                ldc);

hipblasStatus_t hipblasZsyr2kFortran(hipblasHandle_t         handle,
                                     hipblasFillMode_t       uplo,
                                     hipblasOperation_t      transA,
                                     int                     n,
                                     int                     k,
                                     const hipDoubleComplex* alpha,
                                     const hipDoubleComplex* A,
                                     int                     lda,
                                     const hipDoubleComplex* B,
                                     int                     ldb,
                                     const hipDoubleComplex* beta,
                                     hipDoubleComplex*       C,
                                     int                     ldc);
#else
hipblasStatus_t hipblasCsyr2kFortran(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int                   n,
                                     int                   k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int                   lda,
                                     const hipblasComplex* B,
                                     int                   ldb,
                                     const hipblasComplex* beta,
                                     hipblasComplex*       C,
                                     int                   ldc);

hipblasStatus_t hipblasZsyr2kFortran(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int                         lda,
                                     const hipblasDoubleComplex* B,
                                     int                         ldb,
                                     const hipblasDoubleComplex* beta,
                                     hipblasDoubleComplex*       C,
                                     int                         ldc);
#endif

// syr2k_batched
hipblasStatus_t hipblasSsyr2kBatchedFortran(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float* const A[],
                                            int                lda,
                                            const float* const B[],
                                            int                ldb,
                                            const float*       beta,
                                            float* const       C[],
                                            int                ldc,
                                            int                batchCount);

hipblasStatus_t hipblasDsyr2kBatchedFortran(hipblasHandle_t     handle,
                                            hipblasFillMode_t   uplo,
                                            hipblasOperation_t  transA,
                                            int                 n,
                                            int                 k,
                                            const double*       alpha,
                                            const double* const A[],
                                            int                 lda,
                                            const double* const B[],
                                            int                 ldb,
                                            const double*       beta,
                                            double* const       C[],
                                            int                 ldc,
                                            int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyr2kBatchedFortran(hipblasHandle_t         handle,
                                            hipblasFillMode_t       uplo,
                                            hipblasOperation_t      transA,
                                            int                     n,
                                            int                     k,
                                            const hipComplex*       alpha,
                                            const hipComplex* const A[],
                                            int                     lda,
                                            const hipComplex* const B[],
                                            int                     ldb,
                                            const hipComplex*       beta,
                                            hipComplex* const       C[],
                                            int                     ldc,
                                            int                     batchCount);

hipblasStatus_t hipblasZsyr2kBatchedFortran(hipblasHandle_t               handle,
                                            hipblasFillMode_t             uplo,
                                            hipblasOperation_t            transA,
                                            int                           n,
                                            int                           k,
                                            const hipDoubleComplex*       alpha,
                                            const hipDoubleComplex* const A[],
                                            int                           lda,
                                            const hipDoubleComplex* const B[],
                                            int                           ldb,
                                            const hipDoubleComplex*       beta,
                                            hipDoubleComplex* const       C[],
                                            int                           ldc,
                                            int                           batchCount);
#else
hipblasStatus_t hipblasCsyr2kBatchedFortran(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int                         lda,
                                            const hipblasComplex* const B[],
                                            int                         ldb,
                                            const hipblasComplex*       beta,
                                            hipblasComplex* const       C[],
                                            int                         ldc,
                                            int                         batchCount);

hipblasStatus_t hipblasZsyr2kBatchedFortran(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int                               n,
                                            int                               k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int                               lda,
                                            const hipblasDoubleComplex* const B[],
                                            int                               ldb,
                                            const hipblasDoubleComplex*       beta,
                                            hipblasDoubleComplex* const       C[],
                                            int                               ldc,
                                            int                               batchCount);
#endif

// syr2k_strided_batched
hipblasStatus_t hipblasSsyr2kStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const float*       alpha,
                                                   const float*       A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const float*       B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const float*       beta,
                                                   float*             C,
                                                   int                ldc,
                                                   hipblasStride      strideC,
                                                   int                batchCount);

hipblasStatus_t hipblasDsyr2kStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const double*      alpha,
                                                   const double*      A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const double*      B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const double*      beta,
                                                   double*            C,
                                                   int                ldc,
                                                   hipblasStride      strideC,
                                                   int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyr2kStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const hipComplex*  alpha,
                                                   const hipComplex*  A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const hipComplex*  B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const hipComplex*  beta,
                                                   hipComplex*        C,
                                                   int                ldc,
                                                   hipblasStride      strideC,
                                                   int                batchCount);

hipblasStatus_t hipblasZsyr2kStridedBatchedFortran(hipblasHandle_t         handle,
                                                   hipblasFillMode_t       uplo,
                                                   hipblasOperation_t      transA,
                                                   int                     n,
                                                   int                     k,
                                                   const hipDoubleComplex* alpha,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   hipblasStride           strideA,
                                                   const hipDoubleComplex* B,
                                                   int                     ldb,
                                                   hipblasStride           strideB,
                                                   const hipDoubleComplex* beta,
                                                   hipDoubleComplex*       C,
                                                   int                     ldc,
                                                   hipblasStride           strideC,
                                                   int                     batchCount);
#else
hipblasStatus_t hipblasCsyr2kStridedBatchedFortran(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int                   n,
                                                   int                   k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int                   lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int                   ldb,
                                                   hipblasStride         strideB,
                                                   const hipblasComplex* beta,
                                                   hipblasComplex*       C,
                                                   int                   ldc,
                                                   hipblasStride         strideC,
                                                   int                   batchCount);

hipblasStatus_t hipblasZsyr2kStridedBatchedFortran(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int                         lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int                         ldb,
                                                   hipblasStride               strideB,
                                                   const hipblasDoubleComplex* beta,
                                                   hipblasDoubleComplex*       C,
                                                   int                         ldc,
                                                   hipblasStride               strideC,
                                                   int                         batchCount);
#endif

// syrkx
hipblasStatus_t hipblasSsyrkxFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const float*       alpha,
                                     const float*       A,
                                     int                lda,
                                     const float*       B,
                                     int                ldb,
                                     const float*       beta,
                                     float*             C,
                                     int                ldc);

hipblasStatus_t hipblasDsyrkxFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const double*      alpha,
                                     const double*      A,
                                     int                lda,
                                     const double*      B,
                                     int                ldb,
                                     const double*      beta,
                                     double*            C,
                                     int                ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrkxFortran(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     int                n,
                                     int                k,
                                     const hipComplex*  alpha,
                                     const hipComplex*  A,
                                     int                lda,
                                     const hipComplex*  B,
                                     int                ldb,
                                     const hipComplex*  beta,
                                     hipComplex*        C,
                                     int                ldc);

hipblasStatus_t hipblasZsyrkxFortran(hipblasHandle_t         handle,
                                     hipblasFillMode_t       uplo,
                                     hipblasOperation_t      transA,
                                     int                     n,
                                     int                     k,
                                     const hipDoubleComplex* alpha,
                                     const hipDoubleComplex* A,
                                     int                     lda,
                                     const hipDoubleComplex* B,
                                     int                     ldb,
                                     const hipDoubleComplex* beta,
                                     hipDoubleComplex*       C,
                                     int                     ldc);
#else
hipblasStatus_t hipblasCsyrkxFortran(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int                   n,
                                     int                   k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int                   lda,
                                     const hipblasComplex* B,
                                     int                   ldb,
                                     const hipblasComplex* beta,
                                     hipblasComplex*       C,
                                     int                   ldc);

hipblasStatus_t hipblasZsyrkxFortran(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int                         n,
                                     int                         k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int                         lda,
                                     const hipblasDoubleComplex* B,
                                     int                         ldb,
                                     const hipblasDoubleComplex* beta,
                                     hipblasDoubleComplex*       C,
                                     int                         ldc);
#endif

// syrkx_batched
hipblasStatus_t hipblasSsyrkxBatchedFortran(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float* const A[],
                                            int                lda,
                                            const float* const B[],
                                            int                ldb,
                                            const float*       beta,
                                            float* const       C[],
                                            int                ldc,
                                            int                batchCount);

hipblasStatus_t hipblasDsyrkxBatchedFortran(hipblasHandle_t     handle,
                                            hipblasFillMode_t   uplo,
                                            hipblasOperation_t  transA,
                                            int                 n,
                                            int                 k,
                                            const double*       alpha,
                                            const double* const A[],
                                            int                 lda,
                                            const double* const B[],
                                            int                 ldb,
                                            const double*       beta,
                                            double* const       C[],
                                            int                 ldc,
                                            int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrkxBatchedFortran(hipblasHandle_t         handle,
                                            hipblasFillMode_t       uplo,
                                            hipblasOperation_t      transA,
                                            int                     n,
                                            int                     k,
                                            const hipComplex*       alpha,
                                            const hipComplex* const A[],
                                            int                     lda,
                                            const hipComplex* const B[],
                                            int                     ldb,
                                            const hipComplex*       beta,
                                            hipComplex* const       C[],
                                            int                     ldc,
                                            int                     batchCount);

hipblasStatus_t hipblasZsyrkxBatchedFortran(hipblasHandle_t               handle,
                                            hipblasFillMode_t             uplo,
                                            hipblasOperation_t            transA,
                                            int                           n,
                                            int                           k,
                                            const hipDoubleComplex*       alpha,
                                            const hipDoubleComplex* const A[],
                                            int                           lda,
                                            const hipDoubleComplex* const B[],
                                            int                           ldb,
                                            const hipDoubleComplex*       beta,
                                            hipDoubleComplex* const       C[],
                                            int                           ldc,
                                            int                           batchCount);
#else
hipblasStatus_t hipblasCsyrkxBatchedFortran(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int                         lda,
                                            const hipblasComplex* const B[],
                                            int                         ldb,
                                            const hipblasComplex*       beta,
                                            hipblasComplex* const       C[],
                                            int                         ldc,
                                            int                         batchCount);

hipblasStatus_t hipblasZsyrkxBatchedFortran(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int                               n,
                                            int                               k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int                               lda,
                                            const hipblasDoubleComplex* const B[],
                                            int                               ldb,
                                            const hipblasDoubleComplex*       beta,
                                            hipblasDoubleComplex* const       C[],
                                            int                               ldc,
                                            int                               batchCount);
#endif

// syrkx_strided_batched
hipblasStatus_t hipblasSsyrkxStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const float*       alpha,
                                                   const float*       A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const float*       B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const float*       beta,
                                                   float*             C,
                                                   int                ldc,
                                                   hipblasStride      stridec,
                                                   int                batchCount);

hipblasStatus_t hipblasDsyrkxStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const double*      alpha,
                                                   const double*      A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const double*      B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const double*      beta,
                                                   double*            C,
                                                   int                ldc,
                                                   hipblasStride      stridec,
                                                   int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCsyrkxStridedBatchedFortran(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const hipComplex*  alpha,
                                                   const hipComplex*  A,
                                                   int                lda,
                                                   hipblasStride      strideA,
                                                   const hipComplex*  B,
                                                   int                ldb,
                                                   hipblasStride      strideB,
                                                   const hipComplex*  beta,
                                                   hipComplex*        C,
                                                   int                ldc,
                                                   hipblasStride      stridec,
                                                   int                batchCount);

hipblasStatus_t hipblasZsyrkxStridedBatchedFortran(hipblasHandle_t         handle,
                                                   hipblasFillMode_t       uplo,
                                                   hipblasOperation_t      transA,
                                                   int                     n,
                                                   int                     k,
                                                   const hipDoubleComplex* alpha,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   hipblasStride           strideA,
                                                   const hipDoubleComplex* B,
                                                   int                     ldb,
                                                   hipblasStride           strideB,
                                                   const hipDoubleComplex* beta,
                                                   hipDoubleComplex*       C,
                                                   int                     ldc,
                                                   hipblasStride           stridec,
                                                   int                     batchCount);
#else
hipblasStatus_t hipblasCsyrkxStridedBatchedFortran(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int                   n,
                                                   int                   k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int                   lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int                   ldb,
                                                   hipblasStride         strideB,
                                                   const hipblasComplex* beta,
                                                   hipblasComplex*       C,
                                                   int                   ldc,
                                                   hipblasStride         stridec,
                                                   int                   batchCount);

hipblasStatus_t hipblasZsyrkxStridedBatchedFortran(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int                         lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int                         ldb,
                                                   hipblasStride               strideB,
                                                   const hipblasDoubleComplex* beta,
                                                   hipblasDoubleComplex*       C,
                                                   int                         ldc,
                                                   hipblasStride               stridec,
                                                   int                         batchCount);
#endif

// geam
hipblasStatus_t hipblasSgeamFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    const float*       beta,
                                    const float*       B,
                                    int                ldb,
                                    float*             C,
                                    int                ldc);

hipblasStatus_t hipblasDgeamFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      beta,
                                    const double*      B,
                                    int                ldb,
                                    double*            C,
                                    int                ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeamFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const hipComplex*  beta,
                                    const hipComplex*  B,
                                    int                ldb,
                                    hipComplex*        C,
                                    int                ldc);

hipblasStatus_t hipblasZgeamFortran(hipblasHandle_t         handle,
                                    hipblasOperation_t      transa,
                                    hipblasOperation_t      transb,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* beta,
                                    const hipDoubleComplex* B,
                                    int                     ldb,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCgeamFortran(hipblasHandle_t       handle,
                                    hipblasOperation_t    transa,
                                    hipblasOperation_t    transb,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* beta,
                                    const hipblasComplex* B,
                                    int                   ldb,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZgeamFortran(hipblasHandle_t             handle,
                                    hipblasOperation_t          transa,
                                    hipblasOperation_t          transb,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* beta,
                                    const hipblasDoubleComplex* B,
                                    int                         ldb,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// geam_batched
hipblasStatus_t hipblasSgeamBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float*       beta,
                                           const float* const B[],
                                           int                ldb,
                                           float* const       C[],
                                           int                ldc,
                                           int                batchCount);

hipblasStatus_t hipblasDgeamBatchedFortran(hipblasHandle_t     handle,
                                           hipblasOperation_t  transa,
                                           hipblasOperation_t  transb,
                                           int                 m,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double*       beta,
                                           const double* const B[],
                                           int                 ldb,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeamBatchedFortran(hipblasHandle_t         handle,
                                           hipblasOperation_t      transa,
                                           hipblasOperation_t      transb,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex*       beta,
                                           const hipComplex* const B[],
                                           int                     ldb,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZgeamBatchedFortran(hipblasHandle_t               handle,
                                           hipblasOperation_t            transa,
                                           hipblasOperation_t            transb,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex*       beta,
                                           const hipDoubleComplex* const B[],
                                           int                           ldb,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCgeamBatchedFortran(hipblasHandle_t             handle,
                                           hipblasOperation_t          transa,
                                           hipblasOperation_t          transb,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex*       beta,
                                           const hipblasComplex* const B[],
                                           int                         ldb,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZgeamBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transa,
                                           hipblasOperation_t                transb,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex*       beta,
                                           const hipblasDoubleComplex* const B[],
                                           int                               ldb,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// geam_strided_batched
hipblasStatus_t hipblasSgeamStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const float*       beta,
                                                  const float*       B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  float*             C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasDgeamStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const double*      beta,
                                                  const double*      B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  double*            C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeamStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const hipComplex*  beta,
                                                  const hipComplex*  B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  hipComplex*        C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasZgeamStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasOperation_t      transa,
                                                  hipblasOperation_t      transb,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* beta,
                                                  const hipDoubleComplex* B,
                                                  int                     ldb,
                                                  hipblasStride           strideB,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           strideC,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCgeamStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transa,
                                                  hipblasOperation_t    transb,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* beta,
                                                  const hipblasComplex* B,
                                                  int                   ldb,
                                                  hipblasStride         strideB,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         strideC,
                                                  int                   batchCount);

hipblasStatus_t hipblasZgeamStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transa,
                                                  hipblasOperation_t          transb,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* beta,
                                                  const hipblasDoubleComplex* B,
                                                  int                         ldb,
                                                  hipblasStride               strideB,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               strideC,
                                                  int                         batchCount);
#endif

// hemm
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChemmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    hipblasFillMode_t uplo,
                                    int               n,
                                    int               k,
                                    const hipComplex* alpha,
                                    const hipComplex* A,
                                    int               lda,
                                    const hipComplex* B,
                                    int               ldb,
                                    const hipComplex* beta,
                                    hipComplex*       C,
                                    int               ldc);

hipblasStatus_t hipblasZhemmFortran(hipblasHandle_t         handle,
                                    hipblasSideMode_t       side,
                                    hipblasFillMode_t       uplo,
                                    int                     n,
                                    int                     k,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* B,
                                    int                     ldb,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasChemmFortran(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    int                   n,
                                    int                   k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* B,
                                    int                   ldb,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZhemmFortran(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    int                         k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* B,
                                    int                         ldb,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// hemm_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChemmBatchedFortran(hipblasHandle_t         handle,
                                           hipblasSideMode_t       side,
                                           hipblasFillMode_t       uplo,
                                           int                     n,
                                           int                     k,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const B[],
                                           int                     ldb,
                                           const hipComplex*       beta,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZhemmBatchedFortran(hipblasHandle_t               handle,
                                           hipblasSideMode_t             side,
                                           hipblasFillMode_t             uplo,
                                           int                           n,
                                           int                           k,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const B[],
                                           int                           ldb,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasChemmBatchedFortran(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const B[],
                                           int                         ldb,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZhemmBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           int                               n,
                                           int                               k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const B[],
                                           int                               ldb,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// hemm_strided_batched
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasChemmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  hipblasFillMode_t uplo,
                                                  int               n,
                                                  int               k,
                                                  const hipComplex* alpha,
                                                  const hipComplex* A,
                                                  int               lda,
                                                  hipblasStride     strideA,
                                                  const hipComplex* B,
                                                  int               ldb,
                                                  hipblasStride     strideB,
                                                  const hipComplex* beta,
                                                  hipComplex*       C,
                                                  int               ldc,
                                                  hipblasStride     strideC,
                                                  int               batchCount);

hipblasStatus_t hipblasZhemmStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasSideMode_t       side,
                                                  hipblasFillMode_t       uplo,
                                                  int                     n,
                                                  int                     k,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* B,
                                                  int                     ldb,
                                                  hipblasStride           strideB,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           strideC,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasChemmStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  int                   k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* B,
                                                  int                   ldb,
                                                  hipblasStride         strideB,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         strideC,
                                                  int                   batchCount);

hipblasStatus_t hipblasZhemmStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  int                         k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* B,
                                                  int                         ldb,
                                                  hipblasStride               strideB,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               strideC,
                                                  int                         batchCount);
#endif

// trmm
hipblasStatus_t hipblasStrmmFortran(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    const float*       B,
                                    int                ldb,
                                    float*             C,
                                    int                ldc);

hipblasStatus_t hipblasDtrmmFortran(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      B,
                                    int                ldb,
                                    double*            C,
                                    int                ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrmmFortran(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const hipComplex*  B,
                                    int                ldb,
                                    hipComplex*        C,
                                    int                ldc);

hipblasStatus_t hipblasZtrmmFortran(hipblasHandle_t         handle,
                                    hipblasSideMode_t       side,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* B,
                                    int                     ldb,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCtrmmFortran(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* B,
                                    int                   ldb,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZtrmmFortran(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* B,
                                    int                         ldb,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// trmm_batched
hipblasStatus_t hipblasStrmmBatchedFortran(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const B[],
                                           int                ldb,
                                           float* const       C[],
                                           int                ldc,
                                           int                batchCount);

hipblasStatus_t hipblasDtrmmBatchedFortran(hipblasHandle_t     handle,
                                           hipblasSideMode_t   side,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const B[],
                                           int                 ldb,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrmmBatchedFortran(hipblasHandle_t         handle,
                                           hipblasSideMode_t       side,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const B[],
                                           int                     ldb,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZtrmmBatchedFortran(hipblasHandle_t               handle,
                                           hipblasSideMode_t             side,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const B[],
                                           int                           ldb,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCtrmmBatchedFortran(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const B[],
                                           int                         ldb,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZtrmmBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const B[],
                                           int                               ldb,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// trmm_strided_batched
hipblasStatus_t hipblasStrmmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const float*       B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  float*             C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasDtrmmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const double*      B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  double*            C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrmmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const hipComplex*  B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  hipComplex*        C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

hipblasStatus_t hipblasZtrmmStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasSideMode_t       side,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  const hipDoubleComplex* B,
                                                  int                     ldb,
                                                  hipblasStride           strideB,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           strideC,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCtrmmStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* B,
                                                  int                   ldb,
                                                  hipblasStride         strideB,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         strideC,
                                                  int                   batchCount);

hipblasStatus_t hipblasZtrmmStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* B,
                                                  int                         ldb,
                                                  hipblasStride               strideB,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               strideC,
                                                  int                         batchCount);
#endif

// trtri
hipblasStatus_t hipblasStrtriFortran(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const float*      A,
                                     int               lda,
                                     float*            invA,
                                     int               ldinvA);

hipblasStatus_t hipblasDtrtriFortran(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const double*     A,
                                     int               lda,
                                     double*           invA,
                                     int               ldinvA);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrtriFortran(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const hipComplex* A,
                                     int               lda,
                                     hipComplex*       invA,
                                     int               ldinvA);

hipblasStatus_t hipblasZtrtriFortran(hipblasHandle_t         handle,
                                     hipblasFillMode_t       uplo,
                                     hipblasDiagType_t       diag,
                                     int                     n,
                                     const hipDoubleComplex* A,
                                     int                     lda,
                                     hipDoubleComplex*       invA,
                                     int                     ldinvA);
#else
hipblasStatus_t hipblasCtrtriFortran(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasDiagType_t     diag,
                                     int                   n,
                                     const hipblasComplex* A,
                                     int                   lda,
                                     hipblasComplex*       invA,
                                     int                   ldinvA);

hipblasStatus_t hipblasZtrtriFortran(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasDiagType_t           diag,
                                     int                         n,
                                     const hipblasDoubleComplex* A,
                                     int                         lda,
                                     hipblasDoubleComplex*       invA,
                                     int                         ldinvA);
#endif

// trtri_batched
hipblasStatus_t hipblasStrtriBatchedFortran(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasDiagType_t  diag,
                                            int                n,
                                            const float* const A[],
                                            int                lda,
                                            float*             invA[],
                                            int                ldinvA,
                                            int                batch_count);

hipblasStatus_t hipblasDtrtriBatchedFortran(hipblasHandle_t     handle,
                                            hipblasFillMode_t   uplo,
                                            hipblasDiagType_t   diag,
                                            int                 n,
                                            const double* const A[],
                                            int                 lda,
                                            double*             invA[],
                                            int                 ldinvA,
                                            int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrtriBatchedFortran(hipblasHandle_t         handle,
                                            hipblasFillMode_t       uplo,
                                            hipblasDiagType_t       diag,
                                            int                     n,
                                            const hipComplex* const A[],
                                            int                     lda,
                                            hipComplex*             invA[],
                                            int                     ldinvA,
                                            int                     batch_count);

hipblasStatus_t hipblasZtrtriBatchedFortran(hipblasHandle_t               handle,
                                            hipblasFillMode_t             uplo,
                                            hipblasDiagType_t             diag,
                                            int                           n,
                                            const hipDoubleComplex* const A[],
                                            int                           lda,
                                            hipDoubleComplex*             invA[],
                                            int                           ldinvA,
                                            int                           batch_count);
#else
hipblasStatus_t hipblasCtrtriBatchedFortran(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasDiagType_t           diag,
                                            int                         n,
                                            const hipblasComplex* const A[],
                                            int                         lda,
                                            hipblasComplex*             invA[],
                                            int                         ldinvA,
                                            int                         batch_count);

hipblasStatus_t hipblasZtrtriBatchedFortran(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasDiagType_t                 diag,
                                            int                               n,
                                            const hipblasDoubleComplex* const A[],
                                            int                               lda,
                                            hipblasDoubleComplex*             invA[],
                                            int                               ldinvA,
                                            int                               batch_count);
#endif

// trtri_strided_batched
hipblasStatus_t hipblasStrtriStridedBatchedFortran(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const float*      A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   float*            invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count);

hipblasStatus_t hipblasDtrtriStridedBatchedFortran(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const double*     A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   double*           invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrtriStridedBatchedFortran(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const hipComplex* A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   hipComplex*       invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count);

hipblasStatus_t hipblasZtrtriStridedBatchedFortran(hipblasHandle_t         handle,
                                                   hipblasFillMode_t       uplo,
                                                   hipblasDiagType_t       diag,
                                                   int                     n,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   hipblasStride           stride_A,
                                                   hipDoubleComplex*       invA,
                                                   int                     ldinvA,
                                                   hipblasStride           stride_invA,
                                                   int                     batch_count);
#else
hipblasStatus_t hipblasCtrtriStridedBatchedFortran(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasDiagType_t     diag,
                                                   int                   n,
                                                   const hipblasComplex* A,
                                                   int                   lda,
                                                   hipblasStride         stride_A,
                                                   hipblasComplex*       invA,
                                                   int                   ldinvA,
                                                   hipblasStride         stride_invA,
                                                   int                   batch_count);

hipblasStatus_t hipblasZtrtriStridedBatchedFortran(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasDiagType_t           diag,
                                                   int                         n,
                                                   const hipblasDoubleComplex* A,
                                                   int                         lda,
                                                   hipblasStride               stride_A,
                                                   hipblasDoubleComplex*       invA,
                                                   int                         ldinvA,
                                                   hipblasStride               stride_invA,
                                                   int                         batch_count);
#endif

// dgmm
hipblasStatus_t hipblasSdgmmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    int               m,
                                    int               n,
                                    const float*      A,
                                    int               lda,
                                    const float*      x,
                                    int               incx,
                                    float*            C,
                                    int               ldc);

hipblasStatus_t hipblasDdgmmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    int               m,
                                    int               n,
                                    const double*     A,
                                    int               lda,
                                    const double*     x,
                                    int               incx,
                                    double*           C,
                                    int               ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCdgmmFortran(hipblasHandle_t   handle,
                                    hipblasSideMode_t side,
                                    int               m,
                                    int               n,
                                    const hipComplex* A,
                                    int               lda,
                                    const hipComplex* x,
                                    int               incx,
                                    hipComplex*       C,
                                    int               ldc);

hipblasStatus_t hipblasZdgmmFortran(hipblasHandle_t         handle,
                                    hipblasSideMode_t       side,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* x,
                                    int                     incx,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCdgmmFortran(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* x,
                                    int                   incx,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZdgmmFortran(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* x,
                                    int                         incx,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// dgmm_batched
hipblasStatus_t hipblasSdgmmBatchedFortran(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           int                m,
                                           int                n,
                                           const float* const A[],
                                           int                lda,
                                           const float* const x[],
                                           int                incx,
                                           float* const       C[],
                                           int                ldc,
                                           int                batch_count);

hipblasStatus_t hipblasDdgmmBatchedFortran(hipblasHandle_t     handle,
                                           hipblasSideMode_t   side,
                                           int                 m,
                                           int                 n,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const x[],
                                           int                 incx,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCdgmmBatchedFortran(hipblasHandle_t         handle,
                                           hipblasSideMode_t       side,
                                           int                     m,
                                           int                     n,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const x[],
                                           int                     incx,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batch_count);

hipblasStatus_t hipblasZdgmmBatchedFortran(hipblasHandle_t               handle,
                                           hipblasSideMode_t             side,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const x[],
                                           int                           incx,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCdgmmBatchedFortran(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const x[],
                                           int                         incx,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batch_count);

hipblasStatus_t hipblasZdgmmBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const x[],
                                           int                               incx,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batch_count);
#endif

// dgmm_strided_batched
hipblasStatus_t hipblasSdgmmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  int               m,
                                                  int               n,
                                                  const float*      A,
                                                  int               lda,
                                                  hipblasStride     stride_A,
                                                  const float*      x,
                                                  int               incx,
                                                  hipblasStride     stride_x,
                                                  float*            C,
                                                  int               ldc,
                                                  hipblasStride     stride_C,
                                                  int               batch_count);

hipblasStatus_t hipblasDdgmmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  int               m,
                                                  int               n,
                                                  const double*     A,
                                                  int               lda,
                                                  hipblasStride     stride_A,
                                                  const double*     x,
                                                  int               incx,
                                                  hipblasStride     stride_x,
                                                  double*           C,
                                                  int               ldc,
                                                  hipblasStride     stride_C,
                                                  int               batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCdgmmStridedBatchedFortran(hipblasHandle_t   handle,
                                                  hipblasSideMode_t side,
                                                  int               m,
                                                  int               n,
                                                  const hipComplex* A,
                                                  int               lda,
                                                  hipblasStride     stride_A,
                                                  const hipComplex* x,
                                                  int               incx,
                                                  hipblasStride     stride_x,
                                                  hipComplex*       C,
                                                  int               ldc,
                                                  hipblasStride     stride_C,
                                                  int               batch_count);

hipblasStatus_t hipblasZdgmmStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasSideMode_t       side,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           stride_A,
                                                  const hipDoubleComplex* x,
                                                  int                     incx,
                                                  hipblasStride           stride_x,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  hipblasStride           stride_C,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCdgmmStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         stride_A,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasStride         stride_x,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  hipblasStride         stride_C,
                                                  int                   batch_count);

hipblasStatus_t hipblasZdgmmStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               stride_A,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasStride               stride_x,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  hipblasStride               stride_C,
                                                  int                         batch_count);
#endif

// trsm
hipblasStatus_t hipblasStrsmFortran(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    float*             B,
                                    int                ldb);

hipblasStatus_t hipblasDtrsmFortran(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    double*            B,
                                    int                ldb);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrsmFortran(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    hipComplex*        B,
                                    int                ldb);

hipblasStatus_t hipblasZtrsmFortran(hipblasHandle_t         handle,
                                    hipblasSideMode_t       side,
                                    hipblasFillMode_t       uplo,
                                    hipblasOperation_t      transA,
                                    hipblasDiagType_t       diag,
                                    int                     m,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    hipDoubleComplex*       B,
                                    int                     ldb);
#else
hipblasStatus_t hipblasCtrsmFortran(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int                   m,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    hipblasComplex*       B,
                                    int                   ldb);

hipblasStatus_t hipblasZtrsmFortran(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    hipblasDoubleComplex*       B,
                                    int                         ldb);
#endif

// trsm_batched
hipblasStatus_t hipblasStrsmBatchedFortran(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           float* const       B[],
                                           int                ldb,
                                           int                batch_count);

hipblasStatus_t hipblasDtrsmBatchedFortran(hipblasHandle_t     handle,
                                           hipblasSideMode_t   side,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       B[],
                                           int                 ldb,
                                           int                 batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrsmBatchedFortran(hipblasHandle_t         handle,
                                           hipblasSideMode_t       side,
                                           hipblasFillMode_t       uplo,
                                           hipblasOperation_t      transA,
                                           hipblasDiagType_t       diag,
                                           int                     m,
                                           int                     n,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           hipComplex* const       B[],
                                           int                     ldb,
                                           int                     batch_count);

hipblasStatus_t hipblasZtrsmBatchedFortran(hipblasHandle_t               handle,
                                           hipblasSideMode_t             side,
                                           hipblasFillMode_t             uplo,
                                           hipblasOperation_t            transA,
                                           hipblasDiagType_t             diag,
                                           int                           m,
                                           int                           n,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           hipDoubleComplex* const       B[],
                                           int                           ldb,
                                           int                           batch_count);
#else
hipblasStatus_t hipblasCtrsmBatchedFortran(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           hipblasComplex* const       B[],
                                           int                         ldb,
                                           int                         batch_count);

hipblasStatus_t hipblasZtrsmBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int                               m,
                                           int                               n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           hipblasDoubleComplex* const       B[],
                                           int                               ldb,
                                           int                               batch_count);
#endif

// trsm_strided_batched
hipblasStatus_t hipblasStrsmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  float*             B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  int                batch_count);

hipblasStatus_t hipblasDtrsmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  double*            B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCtrsmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  hipComplex*        B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  int                batch_count);

hipblasStatus_t hipblasZtrsmStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasSideMode_t       side,
                                                  hipblasFillMode_t       uplo,
                                                  hipblasOperation_t      transA,
                                                  hipblasDiagType_t       diag,
                                                  int                     m,
                                                  int                     n,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  hipblasStride           strideA,
                                                  hipDoubleComplex*       B,
                                                  int                     ldb,
                                                  hipblasStride           strideB,
                                                  int                     batch_count);
#else
hipblasStatus_t hipblasCtrsmStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  hipblasStride         strideA,
                                                  hipblasComplex*       B,
                                                  int                   ldb,
                                                  hipblasStride         strideB,
                                                  int                   batch_count);

hipblasStatus_t hipblasZtrsmStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasStride               strideA,
                                                  hipblasDoubleComplex*       B,
                                                  int                         ldb,
                                                  hipblasStride               strideB,
                                                  int                         batch_count);
#endif

// gemm
hipblasStatus_t hipblasHgemmFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const hipblasHalf* alpha,
                                    const hipblasHalf* A,
                                    int                lda,
                                    const hipblasHalf* B,
                                    int                ldb,
                                    const hipblasHalf* beta,
                                    hipblasHalf*       C,
                                    int                ldc);

hipblasStatus_t hipblasSgemmFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const float*       alpha,
                                    const float*       A,
                                    int                lda,
                                    const float*       B,
                                    int                ldb,
                                    const float*       beta,
                                    float*             C,
                                    int                ldc);

hipblasStatus_t hipblasDgemmFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      B,
                                    int                ldb,
                                    const double*      beta,
                                    double*            C,
                                    int                ldc);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgemmFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t transa,
                                    hipblasOperation_t transb,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const hipComplex*  alpha,
                                    const hipComplex*  A,
                                    int                lda,
                                    const hipComplex*  B,
                                    int                ldb,
                                    const hipComplex*  beta,
                                    hipComplex*        C,
                                    int                ldc);

hipblasStatus_t hipblasZgemmFortran(hipblasHandle_t         handle,
                                    hipblasOperation_t      transa,
                                    hipblasOperation_t      transb,
                                    int                     m,
                                    int                     n,
                                    int                     k,
                                    const hipDoubleComplex* alpha,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* B,
                                    int                     ldb,
                                    const hipDoubleComplex* beta,
                                    hipDoubleComplex*       C,
                                    int                     ldc);
#else
hipblasStatus_t hipblasCgemmFortran(hipblasHandle_t       handle,
                                    hipblasOperation_t    transa,
                                    hipblasOperation_t    transb,
                                    int                   m,
                                    int                   n,
                                    int                   k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int                   lda,
                                    const hipblasComplex* B,
                                    int                   ldb,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int                   ldc);

hipblasStatus_t hipblasZgemmFortran(hipblasHandle_t             handle,
                                    hipblasOperation_t          transa,
                                    hipblasOperation_t          transb,
                                    int                         m,
                                    int                         n,
                                    int                         k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int                         lda,
                                    const hipblasDoubleComplex* B,
                                    int                         ldb,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int                         ldc);
#endif

// gemm batched
hipblasStatus_t hipblasHgemmBatchedFortran(hipblasHandle_t          handle,
                                           hipblasOperation_t       transa,
                                           hipblasOperation_t       transb,
                                           int                      m,
                                           int                      n,
                                           int                      k,
                                           const hipblasHalf*       alpha,
                                           const hipblasHalf* const A[],
                                           int                      lda,
                                           const hipblasHalf* const B[],
                                           int                      ldb,
                                           const hipblasHalf*       beta,
                                           hipblasHalf* const       C[],
                                           int                      ldc,
                                           int                      batchCount);

hipblasStatus_t hipblasSgemmBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float* const A[],
                                           int                lda,
                                           const float* const B[],
                                           int                ldb,
                                           const float*       beta,
                                           float* const       C[],
                                           int                ldc,
                                           int                batchCount);

hipblasStatus_t hipblasDgemmBatchedFortran(hipblasHandle_t     handle,
                                           hipblasOperation_t  transa,
                                           hipblasOperation_t  transb,
                                           int                 m,
                                           int                 n,
                                           int                 k,
                                           const double*       alpha,
                                           const double* const A[],
                                           int                 lda,
                                           const double* const B[],
                                           int                 ldb,
                                           const double*       beta,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batchCount);
#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgemmBatchedFortran(hipblasHandle_t         handle,
                                           hipblasOperation_t      transa,
                                           hipblasOperation_t      transb,
                                           int                     m,
                                           int                     n,
                                           int                     k,
                                           const hipComplex*       alpha,
                                           const hipComplex* const A[],
                                           int                     lda,
                                           const hipComplex* const B[],
                                           int                     ldb,
                                           const hipComplex*       beta,
                                           hipComplex* const       C[],
                                           int                     ldc,
                                           int                     batchCount);

hipblasStatus_t hipblasZgemmBatchedFortran(hipblasHandle_t               handle,
                                           hipblasOperation_t            transa,
                                           hipblasOperation_t            transb,
                                           int                           m,
                                           int                           n,
                                           int                           k,
                                           const hipDoubleComplex*       alpha,
                                           const hipDoubleComplex* const A[],
                                           int                           lda,
                                           const hipDoubleComplex* const B[],
                                           int                           ldb,
                                           const hipDoubleComplex*       beta,
                                           hipDoubleComplex* const       C[],
                                           int                           ldc,
                                           int                           batchCount);
#else
hipblasStatus_t hipblasCgemmBatchedFortran(hipblasHandle_t             handle,
                                           hipblasOperation_t          transa,
                                           hipblasOperation_t          transb,
                                           int                         m,
                                           int                         n,
                                           int                         k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int                         lda,
                                           const hipblasComplex* const B[],
                                           int                         ldb,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int                         ldc,
                                           int                         batchCount);

hipblasStatus_t hipblasZgemmBatchedFortran(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transa,
                                           hipblasOperation_t                transb,
                                           int                               m,
                                           int                               n,
                                           int                               k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int                               lda,
                                           const hipblasDoubleComplex* const B[],
                                           int                               ldb,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int                               ldc,
                                           int                               batchCount);
#endif

// gemm_strided_batched
hipblasStatus_t hipblasHgemmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const hipblasHalf* alpha,
                                                  const hipblasHalf* A,
                                                  int                lda,
                                                  long long          bsa,
                                                  const hipblasHalf* B,
                                                  int                ldb,
                                                  long long          bsb,
                                                  const hipblasHalf* beta,
                                                  hipblasHalf*       C,
                                                  int                ldc,
                                                  long long          bsc,
                                                  int                batchCount);

hipblasStatus_t hipblasSgemmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const float*       alpha,
                                                  const float*       A,
                                                  int                lda,
                                                  long long          bsa,
                                                  const float*       B,
                                                  int                ldb,
                                                  long long          bsb,
                                                  const float*       beta,
                                                  float*             C,
                                                  int                ldc,
                                                  long long          bsc,
                                                  int                batchCount);

hipblasStatus_t hipblasDgemmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  long long          bsa,
                                                  const double*      B,
                                                  int                ldb,
                                                  long long          bsb,
                                                  const double*      beta,
                                                  double*            C,
                                                  int                ldc,
                                                  long long          bsc,
                                                  int                batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgemmStridedBatchedFortran(hipblasHandle_t    handle,
                                                  hipblasOperation_t transa,
                                                  hipblasOperation_t transb,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const hipComplex*  alpha,
                                                  const hipComplex*  A,
                                                  int                lda,
                                                  long long          bsa,
                                                  const hipComplex*  B,
                                                  int                ldb,
                                                  long long          bsb,
                                                  const hipComplex*  beta,
                                                  hipComplex*        C,
                                                  int                ldc,
                                                  long long          bsc,
                                                  int                batchCount);

hipblasStatus_t hipblasZgemmStridedBatchedFortran(hipblasHandle_t         handle,
                                                  hipblasOperation_t      transa,
                                                  hipblasOperation_t      transb,
                                                  int                     m,
                                                  int                     n,
                                                  int                     k,
                                                  const hipDoubleComplex* alpha,
                                                  const hipDoubleComplex* A,
                                                  int                     lda,
                                                  long long               bsa,
                                                  const hipDoubleComplex* B,
                                                  int                     ldb,
                                                  long long               bsb,
                                                  const hipDoubleComplex* beta,
                                                  hipDoubleComplex*       C,
                                                  int                     ldc,
                                                  long long               bsc,
                                                  int                     batchCount);
#else
hipblasStatus_t hipblasCgemmStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transa,
                                                  hipblasOperation_t    transb,
                                                  int                   m,
                                                  int                   n,
                                                  int                   k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  long long             bsa,
                                                  const hipblasComplex* B,
                                                  int                   ldb,
                                                  long long             bsb,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int                   ldc,
                                                  long long             bsc,
                                                  int                   batchCount);

hipblasStatus_t hipblasZgemmStridedBatchedFortran(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transa,
                                                  hipblasOperation_t          transb,
                                                  int                         m,
                                                  int                         n,
                                                  int                         k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  long long                   bsa,
                                                  const hipblasDoubleComplex* B,
                                                  int                         ldb,
                                                  long long                   bsb,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc,
                                                  long long                   bsc,
                                                  int                         batchCount);
#endif

// gemmex
// If compiling with HIPBLAS_V2, fortran version will accept old interface.
// We aren't testing fortran interface with HIPBLAS_V2 defined, so routing
// fortran to regular C interface in that case to get test code to work.
#ifdef HIPBLAS_V2

#define hipblasGemmExFortran hipblasGemmEx
#define hipblasGemmBatchedExFortran hipblasGemmBatchedEx
#define hipblasGemmStridedBatchedExFortran hipblasGemmStridedBatchedEx

#define hipblasGemmExWithFlagsFortran hipblasGemmExWithFlags
#define hipblasGemmBatchedExWithFlagsFortran hipblasGemmBatchedExWithFlags
#define hipblasGemmStridedBatchedExWithFlagsFortran hipblasGemmStridedBatchedExWithFlags

#else

hipblasStatus_t hipblasGemmExFortran(hipblasHandle_t    handle,
                                     hipblasOperation_t trans_a,
                                     hipblasOperation_t trans_b,
                                     int                m,
                                     int                n,
                                     int                k,
                                     const void*        alpha,
                                     const void*        a,
                                     hipblasDatatype_t  a_type,
                                     int                lda,
                                     const void*        b,
                                     hipblasDatatype_t  b_type,
                                     int                ldb,
                                     const void*        beta,
                                     void*              c,
                                     hipblasDatatype_t  c_type,
                                     int                ldc,
                                     hipblasDatatype_t  compute_type,
                                     hipblasGemmAlgo_t  algo);

hipblasStatus_t hipblasGemmBatchedExFortran(hipblasHandle_t    handle,
                                            hipblasOperation_t trans_a,
                                            hipblasOperation_t trans_b,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const void*        alpha,
                                            const void*        a[],
                                            hipblasDatatype_t  a_type,
                                            int                lda,
                                            const void*        b[],
                                            hipblasDatatype_t  b_type,
                                            int                ldb,
                                            const void*        beta,
                                            void*              c[],
                                            hipblasDatatype_t  c_type,
                                            int                ldc,
                                            int                batch_count,
                                            hipblasDatatype_t  compute_type,
                                            hipblasGemmAlgo_t  algo);

hipblasStatus_t hipblasGemmStridedBatchedExFortran(hipblasHandle_t    handle,
                                                   hipblasOperation_t trans_a,
                                                   hipblasOperation_t trans_b,
                                                   int                m,
                                                   int                n,
                                                   int                k,
                                                   const void*        alpha,
                                                   const void*        a,
                                                   hipblasDatatype_t  a_type,
                                                   int                lda,
                                                   hipblasStride      stride_A,
                                                   const void*        b,
                                                   hipblasDatatype_t  b_type,
                                                   int                ldb,
                                                   hipblasStride      stride_B,
                                                   const void*        beta,
                                                   void*              c,
                                                   hipblasDatatype_t  c_type,
                                                   int                ldc,
                                                   hipblasStride      stride_C,
                                                   int                batch_count,
                                                   hipblasDatatype_t  compute_type,
                                                   hipblasGemmAlgo_t  algo);

hipblasStatus_t hipblasGemmExWithFlagsFortran(hipblasHandle_t    handle,
                                              hipblasOperation_t transA,
                                              hipblasOperation_t transB,
                                              int                m,
                                              int                n,
                                              int                k,
                                              const void*        alpha,
                                              const void*        A,
                                              hipblasDatatype_t  aType,
                                              int                lda,
                                              const void*        B,
                                              hipblasDatatype_t  bType,
                                              int                ldb,
                                              const void*        beta,
                                              void*              C,
                                              hipblasDatatype_t  cType,
                                              int                ldc,
                                              hipblasDatatype_t  computeType,
                                              hipblasGemmAlgo_t  algo,
                                              hipblasGemmFlags_t flags);

hipblasStatus_t hipblasGemmBatchedExWithFlagsFortran(hipblasHandle_t    handle,
                                                     hipblasOperation_t transA,
                                                     hipblasOperation_t transB,
                                                     int                m,
                                                     int                n,
                                                     int                k,
                                                     const void*        alpha,
                                                     const void*        A[],
                                                     hipblasDatatype_t  aType,
                                                     int                lda,
                                                     const void*        B[],
                                                     hipblasDatatype_t  bType,
                                                     int                ldb,
                                                     const void*        beta,
                                                     void*              C[],
                                                     hipblasDatatype_t  cType,
                                                     int                ldc,
                                                     int                batchCount,
                                                     hipblasDatatype_t  computeType,
                                                     hipblasGemmAlgo_t  algo,
                                                     hipblasGemmFlags_t flags);

hipblasStatus_t hipblasGemmStridedBatchedExWithFlagsFortran(hipblasHandle_t    handle,
                                                            hipblasOperation_t transA,
                                                            hipblasOperation_t transB,
                                                            int                m,
                                                            int                n,
                                                            int                k,
                                                            const void*        alpha,
                                                            const void*        A,
                                                            hipblasDatatype_t  aType,
                                                            int                lda,
                                                            hipblasStride      strideA,
                                                            const void*        B,
                                                            hipblasDatatype_t  bType,
                                                            int                ldb,
                                                            hipblasStride      strideB,
                                                            const void*        beta,
                                                            void*              C,
                                                            hipblasDatatype_t  cType,
                                                            int                ldc,
                                                            hipblasStride      strideC,
                                                            int                batchCount,
                                                            hipblasDatatype_t  computeType,
                                                            hipblasGemmAlgo_t  algo,
                                                            hipblasGemmFlags_t flags);

#endif

// trsm_ex
hipblasStatus_t hipblasTrsmExFortran(hipblasHandle_t    handle,
                                     hipblasSideMode_t  side,
                                     hipblasFillMode_t  uplo,
                                     hipblasOperation_t transA,
                                     hipblasDiagType_t  diag,
                                     int                m,
                                     int                n,
                                     const void*        alpha,
                                     void*              A,
                                     int                lda,
                                     void*              B,
                                     int                ldb,
                                     const void*        invA,
                                     int                invA_size,
                                     hipblasDatatype_t  compute_type);

hipblasStatus_t hipblasTrsmBatchedExFortran(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const void*        alpha,
                                            void*              A,
                                            int                lda,
                                            void*              B,
                                            int                ldb,
                                            int                batch_count,
                                            const void*        invA,
                                            int                invA_size,
                                            hipblasDatatype_t  compute_type);

hipblasStatus_t hipblasTrsmStridedBatchedExFortran(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   int                n,
                                                   const void*        alpha,
                                                   void*              A,
                                                   int                lda,
                                                   hipblasStride      stride_A,
                                                   void*              B,
                                                   int                ldb,
                                                   hipblasStride      stride_B,
                                                   int                batch_count,
                                                   const void*        invA,
                                                   int                invA_size,
                                                   hipblasStride      stride_invA,
                                                   hipblasDatatype_t  compute_type);

// // syrk_ex
// hipblasStatus_t hipblasCsyrkExFortran(hipblasHandle_t       handle,
//                                       hipblasFillMode_t     uplo,
//                                       hipblasOperation_t    trans,
//                                       int                   n,
//                                       int                   k,
//                                       const hipblasComplex* alpha,
//                                       const void*           A,
//                                       hipblasDatatype_t     Atype,
//                                       int                   lda,
//                                       const hipblasComplex* beta,
//                                       hipblasComplex*       C,
//                                       hipblasDatatype_t     Ctype,
//                                       int                   ldc);

// // herk_ex
// hipblasStatus_t hipblasCherkExFortran(hipblasHandle_t    handle,
//                                       hipblasFillMode_t  uplo,
//                                       hipblasOperation_t trans,
//                                       int                n,
//                                       int                k,
//                                       const float*       alpha,
//                                       const void*        A,
//                                       hipblasDatatype_t  Atype,
//                                       int                lda,
//                                       const float*       beta,
//                                       hipblasComplex*    C,
//                                       hipblasDatatype_t  Ctype,
//                                       int                ldc);

// axpy_ex
hipblasStatus_t hipblasAxpyExFortran(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       alpha,
                                     hipblasDatatype_t alphaType,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     void*             y,
                                     hipblasDatatype_t yType,
                                     int               incy,
                                     hipblasDatatype_t executionType);

hipblasStatus_t hipblasAxpyBatchedExFortran(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       alpha,
                                            hipblasDatatype_t alphaType,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            void*             y,
                                            hipblasDatatype_t yType,
                                            int               incy,
                                            int               batch_count,
                                            hipblasDatatype_t executionType);

hipblasStatus_t hipblasAxpyStridedBatchedExFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const void*       alpha,
                                                   hipblasDatatype_t alphaType,
                                                   const void*       x,
                                                   hipblasDatatype_t xType,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   void*             y,
                                                   hipblasDatatype_t yType,
                                                   int               incy,
                                                   hipblasStride     stridey,
                                                   int               batch_count,
                                                   hipblasDatatype_t executionType);

// dot_ex
hipblasStatus_t hipblasDotExFortran(hipblasHandle_t   handle,
                                    int               n,
                                    const void*       x,
                                    hipblasDatatype_t xType,
                                    int               incx,
                                    const void*       y,
                                    hipblasDatatype_t yType,
                                    int               incy,
                                    void*             result,
                                    hipblasDatatype_t resultType,
                                    hipblasDatatype_t executionType);

hipblasStatus_t hipblasDotcExFortran(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     const void*       y,
                                     hipblasDatatype_t yType,
                                     int               incy,
                                     void*             result,
                                     hipblasDatatype_t resultType,
                                     hipblasDatatype_t executionType);

hipblasStatus_t hipblasDotBatchedExFortran(hipblasHandle_t   handle,
                                           int               n,
                                           const void*       x,
                                           hipblasDatatype_t xType,
                                           int               incx,
                                           const void*       y,
                                           hipblasDatatype_t yType,
                                           int               incy,
                                           int               batch_count,
                                           void*             result,
                                           hipblasDatatype_t resultType,
                                           hipblasDatatype_t executionType);

hipblasStatus_t hipblasDotcBatchedExFortran(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            const void*       y,
                                            hipblasDatatype_t yType,
                                            int               incy,
                                            int               batch_count,
                                            void*             result,
                                            hipblasDatatype_t resultType,
                                            hipblasDatatype_t executionType);

hipblasStatus_t hipblasDotStridedBatchedExFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  const void*       x,
                                                  hipblasDatatype_t xType,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  const void*       y,
                                                  hipblasDatatype_t yType,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  int               batch_count,
                                                  void*             result,
                                                  hipblasDatatype_t resultType,
                                                  hipblasDatatype_t executionType);

hipblasStatus_t hipblasDotcStridedBatchedExFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const void*       x,
                                                   hipblasDatatype_t xType,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   const void*       y,
                                                   hipblasDatatype_t yType,
                                                   int               incy,
                                                   hipblasStride     stridey,
                                                   int               batch_count,
                                                   void*             result,
                                                   hipblasDatatype_t resultType,
                                                   hipblasDatatype_t executionType);

// nrm2_ex
hipblasStatus_t hipblasNrm2ExFortran(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     void*             result,
                                     hipblasDatatype_t resultType,
                                     hipblasDatatype_t executionType);

hipblasStatus_t hipblasNrm2BatchedExFortran(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            int               batch_count,
                                            void*             result,
                                            hipblasDatatype_t resultType,
                                            hipblasDatatype_t executionType);

hipblasStatus_t hipblasNrm2StridedBatchedExFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const void*       x,
                                                   hipblasDatatype_t xType,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   int               batch_count,
                                                   void*             result,
                                                   hipblasDatatype_t resultType,
                                                   hipblasDatatype_t executionType);

// rot_ex
hipblasStatus_t hipblasRotExFortran(hipblasHandle_t   handle,
                                    int               n,
                                    void*             x,
                                    hipblasDatatype_t xType,
                                    int               incx,
                                    void*             y,
                                    hipblasDatatype_t yType,
                                    int               incy,
                                    const void*       c,
                                    const void*       s,
                                    hipblasDatatype_t csType,
                                    hipblasDatatype_t executionType);

hipblasStatus_t hipblasRotBatchedExFortran(hipblasHandle_t   handle,
                                           int               n,
                                           void*             x,
                                           hipblasDatatype_t xType,
                                           int               incx,
                                           void*             y,
                                           hipblasDatatype_t yType,
                                           int               incy,
                                           const void*       c,
                                           const void*       s,
                                           hipblasDatatype_t csType,
                                           int               batch_count,
                                           hipblasDatatype_t executionType);

hipblasStatus_t hipblasRotStridedBatchedExFortran(hipblasHandle_t   handle,
                                                  int               n,
                                                  void*             x,
                                                  hipblasDatatype_t xType,
                                                  int               incx,
                                                  hipblasStride     stridex,
                                                  void*             y,
                                                  hipblasDatatype_t yType,
                                                  int               incy,
                                                  hipblasStride     stridey,
                                                  const void*       c,
                                                  const void*       s,
                                                  hipblasDatatype_t csType,
                                                  int               batch_count,
                                                  hipblasDatatype_t executionType);

// scal_ex
hipblasStatus_t hipblasScalExFortran(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       alpha,
                                     hipblasDatatype_t alphaType,
                                     void*             x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     hipblasDatatype_t executionType);

hipblasStatus_t hipblasScalBatchedExFortran(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       alpha,
                                            hipblasDatatype_t alphaType,
                                            void*             x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            int               batch_count,
                                            hipblasDatatype_t executionType);

hipblasStatus_t hipblasScalStridedBatchedExFortran(hipblasHandle_t   handle,
                                                   int               n,
                                                   const void*       alpha,
                                                   hipblasDatatype_t alphaType,
                                                   void*             x,
                                                   hipblasDatatype_t xType,
                                                   int               incx,
                                                   hipblasStride     stridex,
                                                   int               batch_count,
                                                   hipblasDatatype_t executionType);

/* ==========
 *    Solver
 * ========== */

// getrf
hipblasStatus_t hipblasSgetrfFortran(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasDgetrfFortran(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetrfFortran(
    hipblasHandle_t handle, const int n, hipComplex* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasZgetrfFortran(
    hipblasHandle_t handle, const int n, hipDoubleComplex* A, const int lda, int* ipiv, int* info);
#else
hipblasStatus_t hipblasCgetrfFortran(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasZgetrfFortran(hipblasHandle_t       handle,
                                     const int             n,
                                     hipblasDoubleComplex* A,
                                     const int             lda,
                                     int*                  ipiv,
                                     int*                  info);
#endif

// getrf_batched
hipblasStatus_t hipblasSgetrfBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgetrfBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetrfBatchedFortran(hipblasHandle_t   handle,
                                            const int         n,
                                            hipComplex* const A[],
                                            const int         lda,
                                            int*              ipiv,
                                            int*              info,
                                            const int         batch_count);

hipblasStatus_t hipblasZgetrfBatchedFortran(hipblasHandle_t         handle,
                                            const int               n,
                                            hipDoubleComplex* const A[],
                                            const int               lda,
                                            int*                    ipiv,
                                            int*                    info,
                                            const int               batch_count);
#else
hipblasStatus_t hipblasCgetrfBatchedFortran(hipblasHandle_t       handle,
                                            const int             n,
                                            hipblasComplex* const A[],
                                            const int             lda,
                                            int*                  ipiv,
                                            int*                  info,
                                            const int             batch_count);

hipblasStatus_t hipblasZgetrfBatchedFortran(hipblasHandle_t             handle,
                                            const int                   n,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            int*                        ipiv,
                                            int*                        info,
                                            const int                   batch_count);
#endif

// getrf_strided_batched
hipblasStatus_t hipblasSgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   float*              A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasDgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   hipComplex*         A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   hipDoubleComplex*   A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);
#else
hipblasStatus_t hipblasCgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   hipblasComplex*     A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgetrfStridedBatchedFortran(hipblasHandle_t       handle,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   const hipblasStride   stride_A,
                                                   int*                  ipiv,
                                                   const hipblasStride   stride_P,
                                                   int*                  info,
                                                   const int             batch_count);
#endif

// getrs
hipblasStatus_t hipblasSgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     float*                   A,
                                     const int                lda,
                                     const int*               ipiv,
                                     float*                   B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasDgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     double*                  A,
                                     const int                lda,
                                     const int*               ipiv,
                                     double*                  B,
                                     const int                ldb,
                                     int*                     info);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipComplex*              A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipComplex*              B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasZgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipDoubleComplex*        A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipDoubleComplex*        B,
                                     const int                ldb,
                                     int*                     info);
#else
hipblasStatus_t hipblasCgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipblasComplex*          A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipblasComplex*          B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasZgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipblasDoubleComplex*    A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipblasDoubleComplex*    B,
                                     const int                ldb,
                                     int*                     info);
#endif

// getrs_batched
hipblasStatus_t hipblasSgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            float* const             A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            float* const             B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasDgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            double* const            A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            double* const            B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipComplex* const        A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            hipComplex* const        B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasZgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipDoubleComplex* const  A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            hipDoubleComplex* const  B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);
#else
hipblasStatus_t hipblasCgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipblasComplex* const    A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            hipblasComplex* const    B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasZgetrsBatchedFortran(hipblasHandle_t             handle,
                                            const hipblasOperation_t    trans,
                                            const int                   n,
                                            const int                   nrhs,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            const int*                  ipiv,
                                            hipblasDoubleComplex* const B[],
                                            const int                   ldb,
                                            int*                        info,
                                            const int                   batch_count);
#endif

// getrs_strided_batched
hipblasStatus_t hipblasSgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   float*                   A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   float*                   B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasDgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   double*                  A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   double*                  B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipComplex*              A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipComplex*              B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasZgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipDoubleComplex*        A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipDoubleComplex*        B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);
#else
hipblasStatus_t hipblasCgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipblasComplex*          A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipblasComplex*          B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasZgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipblasDoubleComplex*    A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipblasDoubleComplex*    B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);
#endif

// getri_batched
hipblasStatus_t hipblasSgetriBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            int*            ipiv,
                                            float* const    C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgetriBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            double* const   C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgetriBatchedFortran(hipblasHandle_t   handle,
                                            const int         n,
                                            hipComplex* const A[],
                                            const int         lda,
                                            int*              ipiv,
                                            hipComplex* const C[],
                                            const int         ldc,
                                            int*              info,
                                            const int         batch_count);

hipblasStatus_t hipblasZgetriBatchedFortran(hipblasHandle_t         handle,
                                            const int               n,
                                            hipDoubleComplex* const A[],
                                            const int               lda,
                                            int*                    ipiv,
                                            hipDoubleComplex* const C[],
                                            const int               ldc,
                                            int*                    info,
                                            const int               batch_count);
#else
hipblasStatus_t hipblasCgetriBatchedFortran(hipblasHandle_t       handle,
                                            const int             n,
                                            hipblasComplex* const A[],
                                            const int             lda,
                                            int*                  ipiv,
                                            hipblasComplex* const C[],
                                            const int             ldc,
                                            int*                  info,
                                            const int             batch_count);

hipblasStatus_t hipblasZgetriBatchedFortran(hipblasHandle_t             handle,
                                            const int                   n,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            int*                        ipiv,
                                            hipblasDoubleComplex* const C[],
                                            const int                   ldc,
                                            int*                        info,
                                            const int                   batch_count);
#endif

// geqrf
hipblasStatus_t hipblasSgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     float*          A,
                                     const int       lda,
                                     float*          tau,
                                     int*            info);

hipblasStatus_t hipblasDgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     double*         A,
                                     const int       lda,
                                     double*         tau,
                                     int*            info);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     hipComplex*     A,
                                     const int       lda,
                                     hipComplex*     tau,
                                     int*            info);

hipblasStatus_t hipblasZgeqrfFortran(hipblasHandle_t   handle,
                                     const int         m,
                                     const int         n,
                                     hipDoubleComplex* A,
                                     const int         lda,
                                     hipDoubleComplex* tau,
                                     int*              info);
#else
hipblasStatus_t hipblasCgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     hipblasComplex* A,
                                     const int       lda,
                                     hipblasComplex* tau,
                                     int*            info);

hipblasStatus_t hipblasZgeqrfFortran(hipblasHandle_t       handle,
                                     const int             m,
                                     const int             n,
                                     hipblasDoubleComplex* A,
                                     const int             lda,
                                     hipblasDoubleComplex* tau,
                                     int*                  info);
#endif

// geqrf_batched
hipblasStatus_t hipblasSgeqrfBatchedFortran(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            float* const    tau[],
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgeqrfBatchedFortran(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            double* const   tau[],
                                            int*            info,
                                            const int       batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeqrfBatchedFortran(hipblasHandle_t   handle,
                                            const int         m,
                                            const int         n,
                                            hipComplex* const A[],
                                            const int         lda,
                                            hipComplex* const tau[],
                                            int*              info,
                                            const int         batch_count);

hipblasStatus_t hipblasZgeqrfBatchedFortran(hipblasHandle_t         handle,
                                            const int               m,
                                            const int               n,
                                            hipDoubleComplex* const A[],
                                            const int               lda,
                                            hipDoubleComplex* const tau[],
                                            int*                    info,
                                            const int               batch_count);
#else
hipblasStatus_t hipblasCgeqrfBatchedFortran(hipblasHandle_t       handle,
                                            const int             m,
                                            const int             n,
                                            hipblasComplex* const A[],
                                            const int             lda,
                                            hipblasComplex* const tau[],
                                            int*                  info,
                                            const int             batch_count);

hipblasStatus_t hipblasZgeqrfBatchedFortran(hipblasHandle_t             handle,
                                            const int                   m,
                                            const int                   n,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            hipblasDoubleComplex* const tau[],
                                            int*                        info,
                                            const int                   batch_count);
#endif

// geqrf_strided_batched
hipblasStatus_t hipblasSgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   float*              A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   float*              tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasDgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   double*             tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   hipComplex*         A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   hipComplex*         tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   hipDoubleComplex*   A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   hipDoubleComplex*   tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);
#else
hipblasStatus_t hipblasCgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   hipblasComplex*     A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   hipblasComplex*     tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgeqrfStridedBatchedFortran(hipblasHandle_t       handle,
                                                   const int             m,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   const hipblasStride   stride_A,
                                                   hipblasDoubleComplex* tau,
                                                   const hipblasStride   stride_T,
                                                   int*                  info,
                                                   const int             batch_count);
#endif

// gels
hipblasStatus_t hipblasSgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    float*             A,
                                    const int          lda,
                                    float*             B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

hipblasStatus_t hipblasDgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    double*            A,
                                    const int          lda,
                                    double*            B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    hipComplex*        A,
                                    const int          lda,
                                    hipComplex*        B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

hipblasStatus_t hipblasZgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    hipDoubleComplex*  A,
                                    const int          lda,
                                    hipDoubleComplex*  B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);
#else
hipblasStatus_t hipblasCgelsFortran(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    hipblasComplex*    A,
                                    const int          lda,
                                    hipblasComplex*    B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo);

hipblasStatus_t hipblasZgelsFortran(hipblasHandle_t       handle,
                                    hipblasOperation_t    trans,
                                    const int             m,
                                    const int             n,
                                    const int             nrhs,
                                    hipblasDoubleComplex* A,
                                    const int             lda,
                                    hipblasDoubleComplex* B,
                                    const int             ldb,
                                    int*                  info,
                                    int*                  deviceInfo);
#endif

// gelsBatched
hipblasStatus_t hipblasSgelsBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           const int          m,
                                           const int          n,
                                           const int          nrhs,
                                           float* const       A[],
                                           const int          lda,
                                           float* const       B[],
                                           const int          ldb,
                                           int*               info,
                                           int*               deviceInfo,
                                           const int          batchCount);

hipblasStatus_t hipblasDgelsBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           const int          m,
                                           const int          n,
                                           const int          nrhs,
                                           double* const      A[],
                                           const int          lda,
                                           double* const      B[],
                                           const int          ldb,
                                           int*               info,
                                           int*               deviceInfo,
                                           const int          batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgelsBatchedFortran(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           const int          m,
                                           const int          n,
                                           const int          nrhs,
                                           hipComplex* const  A[],
                                           const int          lda,
                                           hipComplex* const  B[],
                                           const int          ldb,
                                           int*               info,
                                           int*               deviceInfo,
                                           const int          batchCount);

hipblasStatus_t hipblasZgelsBatchedFortran(hipblasHandle_t         handle,
                                           hipblasOperation_t      trans,
                                           const int               m,
                                           const int               n,
                                           const int               nrhs,
                                           hipDoubleComplex* const A[],
                                           const int               lda,
                                           hipDoubleComplex* const B[],
                                           const int               ldb,
                                           int*                    info,
                                           int*                    deviceInfo,
                                           const int               batchCount);
#else
hipblasStatus_t hipblasCgelsBatchedFortran(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           const int             m,
                                           const int             n,
                                           const int             nrhs,
                                           hipblasComplex* const A[],
                                           const int             lda,
                                           hipblasComplex* const B[],
                                           const int             ldb,
                                           int*                  info,
                                           int*                  deviceInfo,
                                           const int             batchCount);

hipblasStatus_t hipblasZgelsBatchedFortran(hipblasHandle_t             handle,
                                           hipblasOperation_t          trans,
                                           const int                   m,
                                           const int                   n,
                                           const int                   nrhs,
                                           hipblasDoubleComplex* const A[],
                                           const int                   lda,
                                           hipblasDoubleComplex* const B[],
                                           const int                   ldb,
                                           int*                        info,
                                           int*                        deviceInfo,
                                           const int                   batchCount);
#endif

// gelsStridedBatched
hipblasStatus_t hipblasSgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  float*              A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  float*              B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

hipblasStatus_t hipblasDgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  double*             A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  double*             B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

#ifdef HIPBLAS_V2
hipblasStatus_t hipblasCgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  hipComplex*         A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  hipComplex*         B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

hipblasStatus_t hipblasZgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  hipDoubleComplex*   A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  hipDoubleComplex*   B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);
#else
hipblasStatus_t hipblasCgelsStridedBatchedFortran(hipblasHandle_t     handle,
                                                  hipblasOperation_t  trans,
                                                  const int           m,
                                                  const int           n,
                                                  const int           nrhs,
                                                  hipblasComplex*     A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  hipblasComplex*     B,
                                                  const int           ldb,
                                                  const hipblasStride strideB,
                                                  int*                info,
                                                  int*                deviceInfo,
                                                  const int           batchCount);

hipblasStatus_t hipblasZgelsStridedBatchedFortran(hipblasHandle_t       handle,
                                                  hipblasOperation_t    trans,
                                                  const int             m,
                                                  const int             n,
                                                  const int             nrhs,
                                                  hipblasDoubleComplex* A,
                                                  const int             lda,
                                                  const hipblasStride   strideA,
                                                  hipblasDoubleComplex* B,
                                                  const int             ldb,
                                                  const hipblasStride   strideB,
                                                  int*                  info,
                                                  int*                  deviceInfo,
                                                  const int             batchCount);
#endif
}

#endif
