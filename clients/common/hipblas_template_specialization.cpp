/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#include "hipblas.h"
#include "hipblas.hpp"
#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to ROCBLAS C89 interfaces
*/

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// axpy
template <>
hipblasStatus_t hipblasAxpy<hipblasHalf>(hipblasHandle_t    handle,
                                         int                n,
                                         const hipblasHalf* alpha,
                                         const hipblasHalf* x,
                                         int                incx,
                                         hipblasHalf*       y,
                                         int                incy)
{
    return hipblasHaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasAxpy<float>(
    hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasAxpy<double>(hipblasHandle_t handle,
                                    int             n,
                                    const double*   alpha,
                                    const double*   x,
                                    int             incx,
                                    double*         y,
                                    int             incy)
{
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasAxpy<hipblasComplex>(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasComplex*       y,
                                            int                   incy)
{
    return hipblasCaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasAxpy<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy)
{
    return hipblasZaxpy(handle, n, alpha, x, incx, y, incy);
}

// axpy_batched
template <>
hipblasStatus_t hipblasAxpyBatched<hipblasHalf>(hipblasHandle_t          handle,
                                                int                      n,
                                                const hipblasHalf*       alpha,
                                                const hipblasHalf* const x[],
                                                int                      incx,
                                                hipblasHalf* const       y[],
                                                int                      incy,
                                                int                      batch_count)
{
    return hipblasHaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyBatched<float>(hipblasHandle_t    handle,
                                          int                n,
                                          const float*       alpha,
                                          const float* const x[],
                                          int                incx,
                                          float* const       y[],
                                          int                incy,
                                          int                batch_count)
{
    return hipblasSaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyBatched<double>(hipblasHandle_t     handle,
                                           int                 n,
                                           const double*       alpha,
                                           const double* const x[],
                                           int                 incx,
                                           double* const       y[],
                                           int                 incy,
                                           int                 batch_count)
{
    return hipblasDaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batch_count)
{
    return hipblasCaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         int                               n,
                                                         const hipblasDoubleComplex*       alpha,
                                                         const hipblasDoubleComplex* const x[],
                                                         int                               incx,
                                                         hipblasDoubleComplex* const       y[],
                                                         int                               incy,
                                                         int batch_count)
{
    return hipblasZaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count);
}

// axpy_strided_batched
template <>
hipblasStatus_t hipblasAxpyStridedBatched<hipblasHalf>(hipblasHandle_t    handle,
                                                       int                n,
                                                       const hipblasHalf* alpha,
                                                       const hipblasHalf* x,
                                                       int                incx,
                                                       int                stridex,
                                                       hipblasHalf*       y,
                                                       int                incy,
                                                       int                stridey,
                                                       int                batch_count)
{
    return hipblasHaxpyStridedBatched(
        handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyStridedBatched<float>(hipblasHandle_t handle,
                                                 int             n,
                                                 const float*    alpha,
                                                 const float*    x,
                                                 int             incx,
                                                 int             stridex,
                                                 float*          y,
                                                 int             incy,
                                                 int             stridey,
                                                 int             batch_count)
{
    return hipblasSaxpyStridedBatched(
        handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyStridedBatched<double>(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   alpha,
                                                  const double*   x,
                                                  int             incx,
                                                  int             stridex,
                                                  double*         y,
                                                  int             incy,
                                                  int             stridey,
                                                  int             batch_count)
{
    return hipblasDaxpyStridedBatched(
        handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count)
{
    return hipblasCaxpyStridedBatched(
        handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasAxpyStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                hipblasDoubleComplex*       y,
                                                                int                         incy,
                                                                int                         stridey,
                                                                int batch_count)
{
    return hipblasZaxpyStridedBatched(
        handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count);
}

// scal
template <>
hipblasStatus_t
    hipblasScal<float>(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    return hipblasSscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t
    hipblasScal<double>(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
{
    return hipblasDscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipblasComplex>(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx)
{
    return hipblasCscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipblasComplex, float>(
    hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx)
{
    return hipblasCsscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx)
{
    return hipblasZscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipblasDoubleComplex, double>(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx)
{
    return hipblasZdscal(handle, n, alpha, x, incx);
}

// scal_batched
template <>
hipblasStatus_t hipblasScalBatched<float>(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batch_count)
{
    return hipblasSscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<double>(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double* const   x[],
                                           int             incx,
                                           int             batch_count)
{
    return hipblasDscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const hipblasComplex* alpha,
                                                   hipblasComplex* const x[],
                                                   int                   incx,
                                                   int                   batch_count)
{
    return hipblasCscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                         int                         n,
                                                         const hipblasDoubleComplex* alpha,
                                                         hipblasDoubleComplex* const x[],
                                                         int                         incx,
                                                         int                         batch_count)
{
    return hipblasZscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const float*          alpha,
                                                          hipblasComplex* const x[],
                                                          int                   incx,
                                                          int                   batch_count)
{
    return hipblasCsscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                                 int                         n,
                                                                 const double*               alpha,
                                                                 hipblasDoubleComplex* const x[],
                                                                 int                         incx,
                                                                 int batch_count)
{
    return hipblasZdscalBatched(handle, n, alpha, x, incx, batch_count);
}

// scal_strided_batched
template <>
hipblasStatus_t hipblasScalStridedBatched<float>(hipblasHandle_t handle,
                                                 int             n,
                                                 const float*    alpha,
                                                 float*          x,
                                                 int             incx,
                                                 int             stridex,
                                                 int             batch_count)
{
    return hipblasSscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<double>(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   alpha,
                                                  double*         x,
                                                  int             incx,
                                                  int             stridex,
                                                  int             batch_count)
{
    return hipblasDscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          int                   batch_count)
{
    return hipblasCscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                hipblasDoubleComplex*       x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                int batch_count)
{
    return hipblasZscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipblasComplex, float>(hipblasHandle_t handle,
                                                                 int             n,
                                                                 const float*    alpha,
                                                                 hipblasComplex* x,
                                                                 int             incx,
                                                                 int             stridex,
                                                                 int             batch_count)
{
    return hipblasCsscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t handle,
                                                                        int             n,
                                                                        const double*   alpha,
                                                                        hipblasDoubleComplex* x,
                                                                        int                   incx,
                                                                        int stridex,
                                                                        int batch_count)
{
    return hipblasZdscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

//swap
template <>
hipblasStatus_t
    hipblasSwap<float>(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    return hipblasSswap(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t
    hipblasSwap<double>(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy)
{
    return hipblasDswap(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasSwap<hipblasComplex>(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return hipblasCswap(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasSwap<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                  int                   n,
                                                  hipblasDoubleComplex* x,
                                                  int                   incx,
                                                  hipblasDoubleComplex* y,
                                                  int                   incy)
{
    return hipblasZswap(handle, n, x, incx, y, incy);
}

// swap_batched
template <>
hipblasStatus_t hipblasSwapBatched<float>(
    hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batch_count)
{
    return hipblasSswapBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasSwapBatched<double>(
    hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batch_count)
{
    return hipblasDswapBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasSwapBatched<hipblasComplex>(hipblasHandle_t handle,
                                                   int             n,
                                                   hipblasComplex* x[],
                                                   int             incx,
                                                   hipblasComplex* y[],
                                                   int             incy,
                                                   int             batch_count)
{
    return hipblasCswapBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasSwapBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                         int                   n,
                                                         hipblasDoubleComplex* x[],
                                                         int                   incx,
                                                         hipblasDoubleComplex* y[],
                                                         int                   incy,
                                                         int                   batch_count)
{
    return hipblasZswapBatched(handle, n, x, incx, y, incy, batch_count);
}

// swap_strided_batched
template <>
hipblasStatus_t hipblasSwapStridedBatched<float>(hipblasHandle_t handle,
                                                 int             n,
                                                 float*          x,
                                                 int             incx,
                                                 int             stridex,
                                                 float*          y,
                                                 int             incy,
                                                 int             stridey,
                                                 int             batch_count)
{
    return hipblasSswapStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasSwapStridedBatched<double>(hipblasHandle_t handle,
                                                  int             n,
                                                  double*         x,
                                                  int             incx,
                                                  int             stridex,
                                                  double*         y,
                                                  int             incy,
                                                  int             stridey,
                                                  int             batch_count)
{
    return hipblasDswapStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasSwapStridedBatched<hipblasComplex>(hipblasHandle_t handle,
                                                          int             n,
                                                          hipblasComplex* x,
                                                          int             incx,
                                                          int             stridex,
                                                          hipblasComplex* y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batch_count)
{
    return hipblasCswapStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasSwapStridedBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                                int                   n,
                                                                hipblasDoubleComplex* x,
                                                                int                   incx,
                                                                int                   stridex,
                                                                hipblasDoubleComplex* y,
                                                                int                   incy,
                                                                int                   stridey,
                                                                int                   batch_count)
{
    return hipblasZswapStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

// copy
template <>
hipblasStatus_t
    hipblasCopy<float>(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    return hipblasScopy(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasCopy<double>(
    hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
{
    return hipblasDcopy(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasCopy<hipblasComplex>(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return hipblasCcopy(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasCopy<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy)
{
    return hipblasZcopy(handle, n, x, incx, y, incy);
}

// copy_batched
template <>
hipblasStatus_t hipblasCopyBatched<float>(hipblasHandle_t    handle,
                                          int                n,
                                          const float* const x[],
                                          int                incx,
                                          float* const       y[],
                                          int                incy,
                                          int                batch_count)
{
    return hipblasScopyBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasCopyBatched<double>(hipblasHandle_t     handle,
                                           int                 n,
                                           const double* const x[],
                                           int                 incx,
                                           double* const       y[],
                                           int                 incy,
                                           int                 batch_count)
{
    return hipblasDcopyBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasCopyBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batch_count)
{
    return hipblasCcopyBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasCopyBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         int                               n,
                                                         const hipblasDoubleComplex* const x[],
                                                         int                               incx,
                                                         hipblasDoubleComplex* const       y[],
                                                         int                               incy,
                                                         int batch_count)
{
    return hipblasZcopyBatched(handle, n, x, incx, y, incy, batch_count);
}

// copy_strided_batched
template <>
hipblasStatus_t hipblasCopyStridedBatched<float>(hipblasHandle_t handle,
                                                 int             n,
                                                 const float*    x,
                                                 int             incx,
                                                 int             stridex,
                                                 float*          y,
                                                 int             incy,
                                                 int             stridey,
                                                 int             batch_count)
{
    return hipblasScopyStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasCopyStridedBatched<double>(hipblasHandle_t handle,
                                                  int             n,
                                                  const double*   x,
                                                  int             incx,
                                                  int             stridex,
                                                  double*         y,
                                                  int             incy,
                                                  int             stridey,
                                                  int             batch_count)
{
    return hipblasDcopyStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasCopyStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count)
{
    return hipblasCcopyStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasCopyStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                int                         n,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                hipblasDoubleComplex*       y,
                                                                int                         incy,
                                                                int                         stridey,
                                                                int batch_count)
{
    return hipblasZcopyStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

// dot
template <>
hipblasStatus_t hipblasDot<hipblasHalf>(hipblasHandle_t    handle,
                                        int                n,
                                        const hipblasHalf* x,
                                        int                incx,
                                        const hipblasHalf* y,
                                        int                incy,
                                        hipblasHalf*       result)
{
    return hipblasHdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<hipblasBfloat16>(hipblasHandle_t        handle,
                                            int                    n,
                                            const hipblasBfloat16* x,
                                            int                    incx,
                                            const hipblasBfloat16* y,
                                            int                    incy,
                                            hipblasBfloat16*       result)
{
    return hipblasBfdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<float>(hipblasHandle_t handle,
                                  int             n,
                                  const float*    x,
                                  int             incx,
                                  const float*    y,
                                  int             incy,
                                  float*          result)
{
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<double>(hipblasHandle_t handle,
                                   int             n,
                                   const double*   x,
                                   int             incx,
                                   const double*   y,
                                   int             incy,
                                   double*         result)
{
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<hipblasComplex>(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasComplex*       result)
{
    return hipblasCdotu(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                 int                         n,
                                                 const hipblasDoubleComplex* x,
                                                 int                         incx,
                                                 const hipblasDoubleComplex* y,
                                                 int                         incy,
                                                 hipblasDoubleComplex*       result)
{
    return hipblasZdotu(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDotc<hipblasComplex>(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       result)
{
    return hipblasCdotc(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDotc<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasDoubleComplex*       result)
{
    return hipblasZdotc(handle, n, x, incx, y, incy, result);
}

// dot_batched
template <>
hipblasStatus_t hipblasDotBatched<hipblasHalf>(hipblasHandle_t          handle,
                                               int                      n,
                                               const hipblasHalf* const x[],
                                               int                      incx,
                                               const hipblasHalf* const y[],
                                               int                      incy,
                                               int                      batch_count,
                                               hipblasHalf*             result)
{
    return hipblasHdotBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotBatched<hipblasBfloat16>(hipblasHandle_t              handle,
                                                   int                          n,
                                                   const hipblasBfloat16* const x[],
                                                   int                          incx,
                                                   const hipblasBfloat16* const y[],
                                                   int                          incy,
                                                   int                          batch_count,
                                                   hipblasBfloat16*             result)
{
    return hipblasBfdotBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotBatched<float>(hipblasHandle_t    handle,
                                         int                n,
                                         const float* const x[],
                                         int                incx,
                                         const float* const y[],
                                         int                incy,
                                         int                batch_count,
                                         float*             result)
{
    return hipblasSdotBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotBatched<double>(hipblasHandle_t     handle,
                                          int                 n,
                                          const double* const x[],
                                          int                 incx,
                                          const double* const y[],
                                          int                 incy,
                                          int                 batch_count,
                                          double*             result)
{
    return hipblasDdotBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                  int                         n,
                                                  const hipblasComplex* const x[],
                                                  int                         incx,
                                                  const hipblasComplex* const y[],
                                                  int                         incy,
                                                  int                         batch_count,
                                                  hipblasComplex*             result)
{
    return hipblasCdotuBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   int                         batch_count,
                                                   hipblasComplex*             result)
{
    return hipblasCdotcBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                        int                               n,
                                                        const hipblasDoubleComplex* const x[],
                                                        int                               incx,
                                                        const hipblasDoubleComplex* const y[],
                                                        int                               incy,
                                                        int                   batch_count,
                                                        hipblasDoubleComplex* result)
{
    return hipblasZdotuBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         int                               n,
                                                         const hipblasDoubleComplex* const x[],
                                                         int                               incx,
                                                         const hipblasDoubleComplex* const y[],
                                                         int                               incy,
                                                         int                   batch_count,
                                                         hipblasDoubleComplex* result)
{
    return hipblasZdotcBatched(handle, n, x, incx, y, incy, batch_count, result);
}

// dot_strided_batched
template <>
hipblasStatus_t hipblasDotStridedBatched<hipblasHalf>(hipblasHandle_t    handle,
                                                      int                n,
                                                      const hipblasHalf* x,
                                                      int                incx,
                                                      int                stridex,
                                                      const hipblasHalf* y,
                                                      int                incy,
                                                      int                stridey,
                                                      int                batch_count,
                                                      hipblasHalf*       result)
{
    return hipblasHdotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotStridedBatched<hipblasBfloat16>(hipblasHandle_t        handle,
                                                          int                    n,
                                                          const hipblasBfloat16* x,
                                                          int                    incx,
                                                          int                    stridex,
                                                          const hipblasBfloat16* y,
                                                          int                    incy,
                                                          int                    stridey,
                                                          int                    batch_count,
                                                          hipblasBfloat16*       result)
{
    return hipblasBfdotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotStridedBatched<float>(hipblasHandle_t handle,
                                                int             n,
                                                const float*    x,
                                                int             incx,
                                                int             stridex,
                                                const float*    y,
                                                int             incy,
                                                int             stridey,
                                                int             batch_count,
                                                float*          result)
{
    return hipblasSdotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotStridedBatched<double>(hipblasHandle_t handle,
                                                 int             n,
                                                 const double*   x,
                                                 int             incx,
                                                 int             stridex,
                                                 const double*   y,
                                                 int             incy,
                                                 int             stridey,
                                                 int             batch_count,
                                                 double*         result)
{
    return hipblasDdotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                         int                   n,
                                                         const hipblasComplex* x,
                                                         int                   incx,
                                                         int                   stridex,
                                                         const hipblasComplex* y,
                                                         int                   incy,
                                                         int                   stridey,
                                                         int                   batch_count,
                                                         hipblasComplex*       result)
{
    return hipblasCdotuStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count,
                                                          hipblasComplex*       result)
{
    return hipblasCdotcStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                               int                         n,
                                                               const hipblasDoubleComplex* x,
                                                               int                         incx,
                                                               int                         stridex,
                                                               const hipblasDoubleComplex* y,
                                                               int                         incy,
                                                               int                         stridey,
                                                               int                   batch_count,
                                                               hipblasDoubleComplex* result)
{
    return hipblasZdotuStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                int                         n,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                const hipblasDoubleComplex* y,
                                                                int                         incy,
                                                                int                         stridey,
                                                                int                   batch_count,
                                                                hipblasDoubleComplex* result)
{
    return hipblasZdotcStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

// asum
template <>
hipblasStatus_t hipblasAsum<float, float>(
    hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{

    return hipblasSasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasAsum<double, double>(
    hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{

    return hipblasDasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasAsum<hipblasComplex, float>(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{

    return hipblasScasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasAsum<hipblasDoubleComplex, double>(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{

    return hipblasDzasum(handle, n, x, incx, result);
}

// asum_batched
template <>
hipblasStatus_t hipblasAsumBatched<float, float>(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, float* result)
{

    return hipblasSasumBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasAsumBatched<double, double>(hipblasHandle_t     handle,
                                                   int                 n,
                                                   const double* const x[],
                                                   int                 incx,
                                                   int                 batch_count,
                                                   double*             result)
{

    return hipblasDasumBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasAsumBatched<hipblasComplex, float>(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasComplex* const x[],
                                                          int                         incx,
                                                          int                         batch_count,
                                                          float*                      result)
{

    return hipblasScasumBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t
    hipblasAsumBatched<hipblasDoubleComplex, double>(hipblasHandle_t                   handle,
                                                     int                               n,
                                                     const hipblasDoubleComplex* const x[],
                                                     int                               incx,
                                                     int                               batch_count,
                                                     double*                           result)
{

    return hipblasDzasumBatched(handle, n, x, incx, batch_count, result);
}

// asum_strided_batched
template <>
hipblasStatus_t hipblasAsumStridedBatched<float, float>(hipblasHandle_t handle,
                                                        int             n,
                                                        const float*    x,
                                                        int             incx,
                                                        int             stridex,
                                                        int             batch_count,
                                                        float*          result)
{

    return hipblasSasumStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasAsumStridedBatched<double, double>(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batch_count,
                                                          double*         result)
{

    return hipblasDasumStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasAsumStridedBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                                 int                   n,
                                                                 const hipblasComplex* x,
                                                                 int                   incx,
                                                                 int                   stridex,
                                                                 int                   batch_count,
                                                                 float*                result)
{

    return hipblasScasumStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t
    hipblasAsumStridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                            int                         n,
                                                            const hipblasDoubleComplex* x,
                                                            int                         incx,
                                                            int                         stridex,
                                                            int                         batch_count,
                                                            double*                     result)
{

    return hipblasDzasumStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

// nrm2
template <>
hipblasStatus_t hipblasNrm2<float, float>(
    hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{

    return hipblasSnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasNrm2<double, double>(
    hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{

    return hipblasDnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasNrm2<hipblasComplex, float>(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{

    return hipblasScnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasNrm2<hipblasDoubleComplex, double>(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{

    return hipblasDznrm2(handle, n, x, incx, result);
}

// nrm2_batched
template <>
hipblasStatus_t hipblasNrm2Batched<float, float>(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, float* result)
{

    return hipblasSnrm2Batched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasNrm2Batched<double, double>(hipblasHandle_t     handle,
                                                   int                 n,
                                                   const double* const x[],
                                                   int                 incx,
                                                   int                 batch_count,
                                                   double*             result)
{

    return hipblasDnrm2Batched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasNrm2Batched<hipblasComplex, float>(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasComplex* const x[],
                                                          int                         incx,
                                                          int                         batch_count,
                                                          float*                      result)
{

    return hipblasScnrm2Batched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t
    hipblasNrm2Batched<hipblasDoubleComplex, double>(hipblasHandle_t                   handle,
                                                     int                               n,
                                                     const hipblasDoubleComplex* const x[],
                                                     int                               incx,
                                                     int                               batch_count,
                                                     double*                           result)
{

    return hipblasDznrm2Batched(handle, n, x, incx, batch_count, result);
}

// nrm2_strided_batched
template <>
hipblasStatus_t hipblasNrm2StridedBatched<float, float>(hipblasHandle_t handle,
                                                        int             n,
                                                        const float*    x,
                                                        int             incx,
                                                        int             stridex,
                                                        int             batch_count,
                                                        float*          result)
{

    return hipblasSnrm2StridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasNrm2StridedBatched<double, double>(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batch_count,
                                                          double*         result)
{

    return hipblasDnrm2StridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasNrm2StridedBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                                 int                   n,
                                                                 const hipblasComplex* x,
                                                                 int                   incx,
                                                                 int                   stridex,
                                                                 int                   batch_count,
                                                                 float*                result)
{

    return hipblasScnrm2StridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t
    hipblasNrm2StridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                            int                         n,
                                                            const hipblasDoubleComplex* x,
                                                            int                         incx,
                                                            int                         stridex,
                                                            int                         batch_count,
                                                            double*                     result)
{

    return hipblasDznrm2StridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

// rot
template <>
hipblasStatus_t hipblasRot<float>(hipblasHandle_t handle,
                                  int             n,
                                  float*          x,
                                  int             incx,
                                  float*          y,
                                  int             incy,
                                  const float*    c,
                                  const float*    s)
{
    return hipblasSrot(handle, n, x, incx, y, incy, c, s);
}

template <>
hipblasStatus_t hipblasRot<double>(hipblasHandle_t handle,
                                   int             n,
                                   double*         x,
                                   int             incx,
                                   double*         y,
                                   int             incy,
                                   const double*   c,
                                   const double*   s)
{
    return hipblasDrot(handle, n, x, incx, y, incy, c, s);
}

template <>
hipblasStatus_t hipblasRot<hipblasComplex, float>(hipblasHandle_t       handle,
                                                  int                   n,
                                                  hipblasComplex*       x,
                                                  int                   incx,
                                                  hipblasComplex*       y,
                                                  int                   incy,
                                                  const float*          c,
                                                  const hipblasComplex* s)
{
    return hipblasCrot(handle, n, x, incx, y, incy, c, s);
}

template <>
hipblasStatus_t hipblasRot<hipblasComplex, float, float>(hipblasHandle_t handle,
                                                         int             n,
                                                         hipblasComplex* x,
                                                         int             incx,
                                                         hipblasComplex* y,
                                                         int             incy,
                                                         const float*    c,
                                                         const float*    s)
{
    return hipblasCsrot(handle, n, x, incx, y, incy, c, s);
}

template <>
hipblasStatus_t hipblasRot<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                         int                         n,
                                                         hipblasDoubleComplex*       x,
                                                         int                         incx,
                                                         hipblasDoubleComplex*       y,
                                                         int                         incy,
                                                         const double*               c,
                                                         const hipblasDoubleComplex* s)
{
    return hipblasZrot(handle, n, x, incx, y, incy, c, s);
}

template <>
hipblasStatus_t hipblasRot<hipblasDoubleComplex, double, double>(hipblasHandle_t       handle,
                                                                 int                   n,
                                                                 hipblasDoubleComplex* x,
                                                                 int                   incx,
                                                                 hipblasDoubleComplex* y,
                                                                 int                   incy,
                                                                 const double*         c,
                                                                 const double*         s)
{
    return hipblasZdrot(handle, n, x, incx, y, incy, c, s);
}

// rot_batched
template <>
hipblasStatus_t hipblasRotBatched<float>(hipblasHandle_t handle,
                                         int             n,
                                         float* const    x[],
                                         int             incx,
                                         float* const    y[],
                                         int             incy,
                                         const float*    c,
                                         const float*    s,
                                         int             batch_count)
{
    return hipblasSrotBatched(handle, n, x, incx, y, incy, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotBatched<double>(hipblasHandle_t handle,
                                          int             n,
                                          double* const   x[],
                                          int             incx,
                                          double* const   y[],
                                          int             incy,
                                          const double*   c,
                                          const double*   s,
                                          int             batch_count)
{
    return hipblasDrotBatched(handle, n, x, incx, y, incy, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                         int                   n,
                                                         hipblasComplex* const x[],
                                                         int                   incx,
                                                         hipblasComplex* const y[],
                                                         int                   incy,
                                                         const float*          c,
                                                         const hipblasComplex* s,
                                                         int                   batch_count)
{
    return hipblasCrotBatched(handle, n, x, incx, y, incy, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotBatched<hipblasComplex, float, float>(hipblasHandle_t       handle,
                                                                int                   n,
                                                                hipblasComplex* const x[],
                                                                int                   incx,
                                                                hipblasComplex* const y[],
                                                                int                   incy,
                                                                const float*          c,
                                                                const float*          s,
                                                                int                   batch_count)
{
    return hipblasCsrotBatched(handle, n, x, incx, y, incy, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                                int                         n,
                                                                hipblasDoubleComplex* const x[],
                                                                int                         incx,
                                                                hipblasDoubleComplex* const y[],
                                                                int                         incy,
                                                                const double*               c,
                                                                const hipblasDoubleComplex* s,
                                                                int batch_count)
{
    return hipblasZrotBatched(handle, n, x, incx, y, incy, c, s, batch_count);
}

template <>
hipblasStatus_t
    hipblasRotBatched<hipblasDoubleComplex, double, double>(hipblasHandle_t             handle,
                                                            int                         n,
                                                            hipblasDoubleComplex* const x[],
                                                            int                         incx,
                                                            hipblasDoubleComplex* const y[],
                                                            int                         incy,
                                                            const double*               c,
                                                            const double*               s,
                                                            int                         batch_count)
{
    return hipblasZdrotBatched(handle, n, x, incx, y, incy, c, s, batch_count);
}

// rot_strided_batched
template <>
hipblasStatus_t hipblasRotStridedBatched<float>(hipblasHandle_t handle,
                                                int             n,
                                                float*          x,
                                                int             incx,
                                                int             stridex,
                                                float*          y,
                                                int             incy,
                                                int             stridey,
                                                const float*    c,
                                                const float*    s,
                                                int             batch_count)
{
    return hipblasSrotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotStridedBatched<double>(hipblasHandle_t handle,
                                                 int             n,
                                                 double*         x,
                                                 int             incx,
                                                 int             stridex,
                                                 double*         y,
                                                 int             incy,
                                                 int             stridey,
                                                 const double*   c,
                                                 const double*   s,
                                                 int             batch_count)
{
    return hipblasDrotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotStridedBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                                int                   n,
                                                                hipblasComplex*       x,
                                                                int                   incx,
                                                                int                   stridex,
                                                                hipblasComplex*       y,
                                                                int                   incy,
                                                                int                   stridey,
                                                                const float*          c,
                                                                const hipblasComplex* s,
                                                                int                   batch_count)
{
    return hipblasCrotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotStridedBatched<hipblasComplex, float, float>(hipblasHandle_t handle,
                                                                       int             n,
                                                                       hipblasComplex* x,
                                                                       int             incx,
                                                                       int             stridex,
                                                                       hipblasComplex* y,
                                                                       int             incy,
                                                                       int             stridey,
                                                                       const float*    c,
                                                                       const float*    s,
                                                                       int             batch_count)
{
    return hipblasCsrotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count);
}

template <>
hipblasStatus_t
    hipblasRotStridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                           int                         n,
                                                           hipblasDoubleComplex*       x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           hipblasDoubleComplex*       y,
                                                           int                         incy,
                                                           int                         stridey,
                                                           const double*               c,
                                                           const hipblasDoubleComplex* s,
                                                           int                         batch_count)
{
    return hipblasZrotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count);
}

template <>
hipblasStatus_t
    hipblasRotStridedBatched<hipblasDoubleComplex, double, double>(hipblasHandle_t       handle,
                                                                   int                   n,
                                                                   hipblasDoubleComplex* x,
                                                                   int                   incx,
                                                                   int                   stridex,
                                                                   hipblasDoubleComplex* y,
                                                                   int                   incy,
                                                                   int                   stridey,
                                                                   const double*         c,
                                                                   const double*         s,
                                                                   int batch_count)
{
    return hipblasZdrotStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count);
}

// rotg
template <>
hipblasStatus_t hipblasRotg<float>(hipblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    return hipblasSrotg(handle, a, b, c, s);
}

template <>
hipblasStatus_t
    hipblasRotg<double>(hipblasHandle_t handle, double* a, double* b, double* c, double* s)
{
    return hipblasDrotg(handle, a, b, c, s);
}

template <>
hipblasStatus_t hipblasRotg<hipblasComplex, float>(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
{
    return hipblasCrotg(handle, a, b, c, s);
}

template <>
hipblasStatus_t hipblasRotg<hipblasDoubleComplex, double>(hipblasHandle_t       handle,
                                                          hipblasDoubleComplex* a,
                                                          hipblasDoubleComplex* b,
                                                          double*               c,
                                                          hipblasDoubleComplex* s)
{
    return hipblasZrotg(handle, a, b, c, s);
}

// rotg_batched
template <>
hipblasStatus_t hipblasRotgBatched<float>(hipblasHandle_t handle,
                                          float* const    a[],
                                          float* const    b[],
                                          float* const    c[],
                                          float* const    s[],
                                          int             batch_count)
{
    return hipblasSrotgBatched(handle, a, b, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotgBatched<double>(hipblasHandle_t handle,
                                           double* const   a[],
                                           double* const   b[],
                                           double* const   c[],
                                           double* const   s[],
                                           int             batch_count)
{
    return hipblasDrotgBatched(handle, a, b, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotgBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                          hipblasComplex* const a[],
                                                          hipblasComplex* const b[],
                                                          float* const          c[],
                                                          hipblasComplex* const s[],
                                                          int                   batch_count)
{
    return hipblasCrotgBatched(handle, a, b, c, s, batch_count);
}

template <>
hipblasStatus_t hipblasRotgBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                                 hipblasDoubleComplex* const a[],
                                                                 hipblasDoubleComplex* const b[],
                                                                 double* const               c[],
                                                                 hipblasDoubleComplex* const s[],
                                                                 int batch_count)
{
    return hipblasZrotgBatched(handle, a, b, c, s, batch_count);
}

// rotg_strided_batched
template <>
hipblasStatus_t hipblasRotgStridedBatched<float>(hipblasHandle_t handle,
                                                 float*          a,
                                                 int             stridea,
                                                 float*          b,
                                                 int             strideb,
                                                 float*          c,
                                                 int             stridec,
                                                 float*          s,
                                                 int             strides,
                                                 int             batch_count)
{
    return hipblasSrotgStridedBatched(
        handle, a, stridea, b, strideb, c, stridec, s, strides, batch_count);
}

template <>
hipblasStatus_t hipblasRotgStridedBatched<double>(hipblasHandle_t handle,
                                                  double*         a,
                                                  int             stridea,
                                                  double*         b,
                                                  int             strideb,
                                                  double*         c,
                                                  int             stridec,
                                                  double*         s,
                                                  int             strides,
                                                  int             batch_count)
{
    return hipblasDrotgStridedBatched(
        handle, a, stridea, b, strideb, c, stridec, s, strides, batch_count);
}

template <>
hipblasStatus_t hipblasRotgStridedBatched<hipblasComplex, float>(hipblasHandle_t handle,
                                                                 hipblasComplex* a,
                                                                 int             stridea,
                                                                 hipblasComplex* b,
                                                                 int             strideb,
                                                                 float*          c,
                                                                 int             stridec,
                                                                 hipblasComplex* s,
                                                                 int             strides,
                                                                 int             batch_count)
{
    return hipblasCrotgStridedBatched(
        handle, a, stridea, b, strideb, c, stridec, s, strides, batch_count);
}

template <>
hipblasStatus_t hipblasRotgStridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t handle,
                                                                        hipblasDoubleComplex* a,
                                                                        int stridea,
                                                                        hipblasDoubleComplex* b,
                                                                        int     strideb,
                                                                        double* c,
                                                                        int     stridec,
                                                                        hipblasDoubleComplex* s,
                                                                        int strides,
                                                                        int batch_count)
{
    return hipblasZrotgStridedBatched(
        handle, a, stridea, b, strideb, c, stridec, s, strides, batch_count);
}

// rotm
template <>
hipblasStatus_t hipblasRotm<float>(
    hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
{
    return hipblasSrotm(handle, n, x, incx, y, incy, param);
}

template <>
hipblasStatus_t hipblasRotm<double>(
    hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param)
{
    return hipblasDrotm(handle, n, x, incx, y, incy, param);
}

// rotm_batched
template <>
hipblasStatus_t hipblasRotmBatched<float>(hipblasHandle_t    handle,
                                          int                n,
                                          float* const       x[],
                                          int                incx,
                                          float* const       y[],
                                          int                incy,
                                          const float* const param[],
                                          int                batch_count)
{
    return hipblasSrotmBatched(handle, n, x, incx, y, incy, param, batch_count);
}

template <>
hipblasStatus_t hipblasRotmBatched<double>(hipblasHandle_t     handle,
                                           int                 n,
                                           double* const       x[],
                                           int                 incx,
                                           double* const       y[],
                                           int                 incy,
                                           const double* const param[],
                                           int                 batch_count)
{
    return hipblasDrotmBatched(handle, n, x, incx, y, incy, param, batch_count);
}

// rotm_strided_batched
template <>
hipblasStatus_t hipblasRotmStridedBatched<float>(hipblasHandle_t handle,
                                                 int             n,
                                                 float*          x,
                                                 int             incx,
                                                 int             stridex,
                                                 float*          y,
                                                 int             incy,
                                                 int             stridey,
                                                 const float*    param,
                                                 int             strideparam,
                                                 int             batch_count)
{
    return hipblasSrotmStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, param, strideparam, batch_count);
}

template <>
hipblasStatus_t hipblasRotmStridedBatched<double>(hipblasHandle_t handle,
                                                  int             n,
                                                  double*         x,
                                                  int             incx,
                                                  int             stridex,
                                                  double*         y,
                                                  int             incy,
                                                  int             stridey,
                                                  const double*   param,
                                                  int             strideparam,
                                                  int             batch_count)
{
    return hipblasDrotmStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, param, strideparam, batch_count);
}

// rotmg
template <>
hipblasStatus_t hipblasRotmg<float>(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    return hipblasSrotmg(handle, d1, d2, x1, y1, param);
}

template <>
hipblasStatus_t hipblasRotmg<double>(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param)
{
    return hipblasDrotmg(handle, d1, d2, x1, y1, param);
}

// rotmg_batched
template <>
hipblasStatus_t hipblasRotmgBatched<float>(hipblasHandle_t    handle,
                                           float* const       d1[],
                                           float* const       d2[],
                                           float* const       x1[],
                                           const float* const y1[],
                                           float* const       param[],
                                           int                batch_count)
{
    return hipblasSrotmgBatched(handle, d1, d2, x1, y1, param, batch_count);
}

template <>
hipblasStatus_t hipblasRotmgBatched<double>(hipblasHandle_t     handle,
                                            double* const       d1[],
                                            double* const       d2[],
                                            double* const       x1[],
                                            const double* const y1[],
                                            double* const       param[],
                                            int                 batch_count)
{
    return hipblasDrotmgBatched(handle, d1, d2, x1, y1, param, batch_count);
}

// rotmg_strided_batched
template <>
hipblasStatus_t hipblasRotmgStridedBatched<float>(hipblasHandle_t handle,
                                                  float*          d1,
                                                  int             stride_d1,
                                                  float*          d2,
                                                  int             stride_d2,
                                                  float*          x1,
                                                  int             stride_x1,
                                                  const float*    y1,
                                                  int             stride_y1,
                                                  float*          param,
                                                  int             strideparam,
                                                  int             batch_count)
{
    return hipblasSrotmgStridedBatched(handle,
                                       d1,
                                       stride_d1,
                                       d2,
                                       stride_d2,
                                       x1,
                                       stride_x1,
                                       y1,
                                       stride_y1,
                                       param,
                                       strideparam,
                                       batch_count);
}

template <>
hipblasStatus_t hipblasRotmgStridedBatched<double>(hipblasHandle_t handle,
                                                   double*         d1,
                                                   int             stride_d1,
                                                   double*         d2,
                                                   int             stride_d2,
                                                   double*         x1,
                                                   int             stride_x1,
                                                   const double*   y1,
                                                   int             stride_y1,
                                                   double*         param,
                                                   int             strideparam,
                                                   int             batch_count)
{
    return hipblasDrotmgStridedBatched(handle,
                                       d1,
                                       stride_d1,
                                       d2,
                                       stride_d2,
                                       x1,
                                       stride_x1,
                                       y1,
                                       stride_y1,
                                       param,
                                       strideparam,
                                       batch_count);
}

// amax
template <>
hipblasStatus_t
    hipblasIamax<float>(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return hipblasIsamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t
    hipblasIamax<double>(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return hipblasIdamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasIamax<hipblasComplex>(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return hipblasIcamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasIamax<hipblasDoubleComplex>(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamax(handle, n, x, incx, result);
}

// amax_batched
template <>
hipblasStatus_t hipblasIamaxBatched<float>(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result)
{
    return hipblasIsamaxBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasIamaxBatched<double>(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result)
{
    return hipblasIdamaxBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasIamaxBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const hipblasComplex* const x[],
                                                    int                         incx,
                                                    int                         batch_count,
                                                    int*                        result)
{
    return hipblasIcamaxBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasIamaxBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                          int                               n,
                                                          const hipblasDoubleComplex* const x[],
                                                          int                               incx,
                                                          int  batch_count,
                                                          int* result)
{
    return hipblasIzamaxBatched(handle, n, x, incx, batch_count, result);
}

// amax_strided_batched
template <>
hipblasStatus_t hipblasIamaxStridedBatched<float>(hipblasHandle_t handle,
                                                  int             n,
                                                  const float*    x,
                                                  int             incx,
                                                  int             stridex,
                                                  int             batch_count,
                                                  int*            result)
{
    return hipblasIsamaxStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasIamaxStridedBatched<double>(hipblasHandle_t handle,
                                                   int             n,
                                                   const double*   x,
                                                   int             incx,
                                                   int             stridex,
                                                   int             batch_count,
                                                   int*            result)
{
    return hipblasIdamaxStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasIamaxStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batch_count,
                                                           int*                  result)
{
    return hipblasIcamaxStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasIamaxStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                 int                         n,
                                                                 const hipblasDoubleComplex* x,
                                                                 int                         incx,
                                                                 int  stridex,
                                                                 int  batch_count,
                                                                 int* result)
{
    return hipblasIzamaxStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

// amin
template <>
hipblasStatus_t
    hipblasIamin<float>(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return hipblasIsamin(handle, n, x, incx, result);
}

template <>
hipblasStatus_t
    hipblasIamin<double>(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return hipblasIdamin(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasIamin<hipblasComplex>(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return hipblasIcamin(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasIamin<hipblasDoubleComplex>(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamin(handle, n, x, incx, result);
}

// amin_batched
template <>
hipblasStatus_t hipblasIaminBatched<float>(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result)
{
    return hipblasIsaminBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasIaminBatched<double>(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result)
{
    return hipblasIdaminBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasIaminBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const hipblasComplex* const x[],
                                                    int                         incx,
                                                    int                         batch_count,
                                                    int*                        result)
{
    return hipblasIcaminBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasIaminBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                          int                               n,
                                                          const hipblasDoubleComplex* const x[],
                                                          int                               incx,
                                                          int  batch_count,
                                                          int* result)
{
    return hipblasIzaminBatched(handle, n, x, incx, batch_count, result);
}

// amin_strided_batched
template <>
hipblasStatus_t hipblasIaminStridedBatched<float>(hipblasHandle_t handle,
                                                  int             n,
                                                  const float*    x,
                                                  int             incx,
                                                  int             stridex,
                                                  int             batch_count,
                                                  int*            result)
{
    return hipblasIsaminStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasIaminStridedBatched<double>(hipblasHandle_t handle,
                                                   int             n,
                                                   const double*   x,
                                                   int             incx,
                                                   int             stridex,
                                                   int             batch_count,
                                                   int*            result)
{
    return hipblasIdaminStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasIaminStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batch_count,
                                                           int*                  result)
{
    return hipblasIcaminStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasIaminStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                 int                         n,
                                                                 const hipblasDoubleComplex* x,
                                                                 int                         incx,
                                                                 int  stridex,
                                                                 int  batch_count,
                                                                 int* result)
{
    return hipblasIzaminStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
template <>
hipblasStatus_t hipblasGbmv<float>(hipblasHandle_t    handle,
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
                                   int                incy)
{
    return hipblasSgbmv(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGbmv<double>(hipblasHandle_t    handle,
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
                                    int                incy)
{
    return hipblasDgbmv(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGbmv<hipblasComplex>(hipblasHandle_t       handle,
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
                                            int                   incy)
{
    return hipblasCgbmv(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGbmv<hipblasDoubleComplex>(hipblasHandle_t             handle,
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
                                                  int                         incy)
{
    return hipblasZgbmv(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

// gbmv_batched
template <>
hipblasStatus_t hipblasGbmvBatched<float>(hipblasHandle_t    handle,
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
                                          int                batch_count)
{
    return hipblasSgbmvBatched(
        handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGbmvBatched<double>(hipblasHandle_t     handle,
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
                                           int                 batch_count)
{
    return hipblasDgbmvBatched(
        handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGbmvBatched<hipblasComplex>(hipblasHandle_t             handle,
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
                                                   int                         batch_count)
{
    return hipblasCgbmvBatched(
        handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGbmvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
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
                                                         int batch_count)
{
    return hipblasZgbmvBatched(
        handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

// gbmv_strided_batched
template <>
hipblasStatus_t hipblasGbmvStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 int                m,
                                                 int                n,
                                                 int                kl,
                                                 int                ku,
                                                 const float*       alpha,
                                                 const float*       A,
                                                 int                lda,
                                                 int                stride_a,
                                                 const float*       x,
                                                 int                incx,
                                                 int                stride_x,
                                                 const float*       beta,
                                                 float*             y,
                                                 int                incy,
                                                 int                stride_y,
                                                 int                batch_count)
{
    return hipblasSgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      alpha,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stride_x,
                                      beta,
                                      y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGbmvStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  int                kl,
                                                  int                ku,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  int                stride_a,
                                                  const double*      x,
                                                  int                incx,
                                                  int                stride_x,
                                                  const double*      beta,
                                                  double*            y,
                                                  int                incy,
                                                  int                stride_y,
                                                  int                batch_count)
{
    return hipblasDgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      alpha,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stride_x,
                                      beta,
                                      y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGbmvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasOperation_t    transA,
                                                          int                   m,
                                                          int                   n,
                                                          int                   kl,
                                                          int                   ku,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   stride_a,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stride_x,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stride_y,
                                                          int                   batch_count)
{
    return hipblasCgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      alpha,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stride_x,
                                      beta,
                                      y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGbmvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasOperation_t          transA,
                                                                int                         m,
                                                                int                         n,
                                                                int                         kl,
                                                                int                         ku,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int stride_a,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int stride_x,
                                                                const hipblasDoubleComplex* beta,
                                                                hipblasDoubleComplex*       y,
                                                                int                         incy,
                                                                int stride_y,
                                                                int batch_count)
{
    return hipblasZgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      alpha,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stride_x,
                                      beta,
                                      y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

// gemv
template <>
hipblasStatus_t hipblasGemv<float>(hipblasHandle_t    handle,
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
                                   int                incy)
{
    return hipblasSgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGemv<double>(hipblasHandle_t    handle,
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
                                    int                incy)
{
    return hipblasDgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGemv<hipblasComplex>(hipblasHandle_t       handle,
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
                                            int                   incy)
{
    return hipblasCgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGemv<hipblasDoubleComplex>(hipblasHandle_t             handle,
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
                                                  int                         incy)
{
    return hipblasZgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// gemv_batched
template <>
hipblasStatus_t hipblasGemvBatched<float>(hipblasHandle_t    handle,
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
                                          int                batch_count)
{
    return hipblasSgemvBatched(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGemvBatched<double>(hipblasHandle_t     handle,
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
                                           int                 batch_count)
{
    return hipblasDgemvBatched(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGemvBatched<hipblasComplex>(hipblasHandle_t             handle,
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
                                                   int                         batch_count)
{
    return hipblasCgemvBatched(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGemvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
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
                                                         int batch_count)
{
    return hipblasZgemvBatched(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

// gemv_strided_batched
template <>
hipblasStatus_t hipblasGemvStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 int                m,
                                                 int                n,
                                                 const float*       alpha,
                                                 const float*       A,
                                                 int                lda,
                                                 int                strideA,
                                                 const float*       x,
                                                 int                incx,
                                                 int                stridex,
                                                 const float*       beta,
                                                 float*             y,
                                                 int                incy,
                                                 int                stridey,
                                                 int                batch_count)
{
    return hipblasSgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      x,
                                      incx,
                                      stridex,
                                      beta,
                                      y,
                                      incy,
                                      stridey,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemvStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  int                strideA,
                                                  const double*      x,
                                                  int                incx,
                                                  int                stridex,
                                                  const double*      beta,
                                                  double*            y,
                                                  int                incy,
                                                  int                stridey,
                                                  int                batch_count)
{
    return hipblasDgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      x,
                                      incx,
                                      stridex,
                                      beta,
                                      y,
                                      incy,
                                      stridey,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasOperation_t    transA,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   strideA,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count)
{
    return hipblasCgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      x,
                                      incx,
                                      stridex,
                                      beta,
                                      y,
                                      incy,
                                      stridey,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasOperation_t          transA,
                                                                int                         m,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int                         strideA,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                const hipblasDoubleComplex* beta,
                                                                hipblasDoubleComplex*       y,
                                                                int                         incy,
                                                                int                         stridey,
                                                                int batch_count)
{
    return hipblasZgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      x,
                                      incx,
                                      stridex,
                                      beta,
                                      y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// ger
template <>
hipblasStatus_t hipblasGer<float>(hipblasHandle_t handle,
                                  int             m,
                                  int             n,
                                  const float*    alpha,
                                  const float*    x,
                                  int             incx,
                                  const float*    y,
                                  int             incy,
                                  float*          A,
                                  int             lda)
{

    return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasGer<double>(hipblasHandle_t handle,
                                   int             m,
                                   int             n,
                                   const double*   alpha,
                                   const double*   x,
                                   int             incx,
                                   const double*   y,
                                   int             incy,
                                   double*         A,
                                   int             lda)
{

    return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

// ger_batched
template <>
hipblasStatus_t hipblasGerBatched<float>(hipblasHandle_t    handle,
                                         int                m,
                                         int                n,
                                         const float*       alpha,
                                         const float* const x[],
                                         int                incx,
                                         const float* const y[],
                                         int                incy,
                                         float* const       A[],
                                         int                lda,
                                         int                batch_count)
{

    return hipblasSgerBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

template <>
hipblasStatus_t hipblasGerBatched<double>(hipblasHandle_t     handle,
                                          int                 m,
                                          int                 n,
                                          const double*       alpha,
                                          const double* const x[],
                                          int                 incx,
                                          const double* const y[],
                                          int                 incy,
                                          double* const       A[],
                                          int                 lda,
                                          int                 batch_count)
{

    return hipblasDgerBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

// ger_strided_batched
template <>
hipblasStatus_t hipblasGerStridedBatched<float>(hipblasHandle_t handle,
                                                int             m,
                                                int             n,
                                                const float*    alpha,
                                                const float*    x,
                                                int             incx,
                                                int             stridex,
                                                const float*    y,
                                                int             incy,
                                                int             stridey,
                                                float*          A,
                                                int             lda,
                                                int             strideA,
                                                int             batch_count)
{

    return hipblasSgerStridedBatched(
        handle, m, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}

template <>
hipblasStatus_t hipblasGerStridedBatched<double>(hipblasHandle_t handle,
                                                 int             m,
                                                 int             n,
                                                 const double*   alpha,
                                                 const double*   x,
                                                 int             incx,
                                                 int             stridex,
                                                 const double*   y,
                                                 int             incy,
                                                 int             stridey,
                                                 double*         A,
                                                 int             lda,
                                                 int             strideA,
                                                 int             batch_count)
{

    return hipblasDgerStridedBatched(
        handle, m, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}

// syr
template <>
hipblasStatus_t hipblasSyr<float>(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const float*      alpha,
                                  const float*      x,
                                  int               incx,
                                  float*            A,
                                  int               lda)
{
    return hipblasSsyr(handle, uplo, n, alpha, x, incx, A, lda);
}

template <>
hipblasStatus_t hipblasSyr<double>(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const double*     alpha,
                                   const double*     x,
                                   int               incx,
                                   double*           A,
                                   int               lda)
{
    return hipblasDsyr(handle, uplo, n, alpha, x, incx, A, lda);
}

// syr_batched
template <>
hipblasStatus_t hipblasSyrBatched<float>(hipblasHandle_t    handle,
                                         hipblasFillMode_t  uplo,
                                         int                n,
                                         const float*       alpha,
                                         const float* const x[],
                                         int                incx,
                                         float* const       A[],
                                         int                lda,
                                         int                batch_count)
{
    return hipblasSsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count);
}

template <>
hipblasStatus_t hipblasSyrBatched<double>(hipblasHandle_t     handle,
                                          hipblasFillMode_t   uplo,
                                          int                 n,
                                          const double*       alpha,
                                          const double* const x[],
                                          int                 incx,
                                          double* const       A[],
                                          int                 lda,
                                          int                 batch_count)
{
    return hipblasDsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count);
}

// syr_strided_batched
template <>
hipblasStatus_t hipblasSyrStridedBatched<float>(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const float*      alpha,
                                                const float*      x,
                                                int               incx,
                                                int               stridex,
                                                float*            A,
                                                int               lda,
                                                int               strideA,
                                                int               batch_count)
{
    return hipblasSsyrStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, A, lda, strideA, batch_count);
}

template <>
hipblasStatus_t hipblasSyrStridedBatched<double>(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const double*     alpha,
                                                 const double*     x,
                                                 int               incx,
                                                 int               stridex,
                                                 double*           A,
                                                 int               lda,
                                                 int               strideA,
                                                 int               batch_count)
{
    return hipblasDsyrStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, A, lda, strideA, batch_count);
}

// trsv
template <>
hipblasStatus_t hipblasTrsv<float>(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const float*       A,
                                   int                lda,
                                   float*             x,
                                   int                incx)
{
    return hipblasStrsv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTrsv<double>(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx)
{
    return hipblasDtrsv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

// trsv_batched
template <>
hipblasStatus_t hipblasTrsvBatched<float>(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const float* const A[],
                                          int                lda,
                                          float* const       x[],
                                          int                incx,
                                          int                batch_count)
{
    return hipblasStrsvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTrsvBatched<double>(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       x[],
                                           int                 incx,
                                           int                 batch_count)
{
    return hipblasDtrsvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

// trsv_strided_batched
template <>
hipblasStatus_t hipblasTrsvStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 const float*       A,
                                                 int                lda,
                                                 int                strideA,
                                                 float*             x,
                                                 int                incx,
                                                 int                stridex,
                                                 int                batch_count)
{
    return hipblasStrsvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, strideA, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasTrsvStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const double*      A,
                                                  int                lda,
                                                  int                strideA,
                                                  double*            x,
                                                  int                incx,
                                                  int                stridex,
                                                  int                batch_count)
{
    return hipblasDtrsvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, strideA, x, incx, stridex, batch_count);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

//

/*
    template<>
    hipblasStatus_t
    hipblasTrtri<float>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        float *A, int lda,
        float *invA, int ldinvA){
        return hipblasStrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }

    template<>
    hipblasStatus_t
    hipblasTrtri<double>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        double *A, int lda,
        double *invA, int ldinvA){
        return hipblasDtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }

    template<>
    hipblasStatus_t
    hipblasTrtri_batched<float>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        float *A, int lda, int bsa,
        float *invA, int ldinvA, int bsinvA,
        int batch_count){
        return hipblasStrtri_batched(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA,
   batch_count);
    }

    template<>
    hipblasStatus_t
    hipblasTrtri_batched<double>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        double *A, int lda, int bsa,
        double *invA, int ldinvA, int bsinvA,
        int batch_count){
        return hipblasDtrtri_batched(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA,
   batch_count);
    }
*/

// gemm
template <>
hipblasStatus_t hipblasGemm<hipblasHalf>(hipblasHandle_t    handle,
                                         hipblasOperation_t transA,
                                         hipblasOperation_t transB,
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
                                         int                ldc)
{
    return hipblasHgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
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
                                   int                ldc)
{
    return hipblasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    hipblasOperation_t transB,
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
                                    int                ldc)
{
    return hipblasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasOperation_t    transA,
                                            hipblasOperation_t    transB,
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
                                            int                   ldc)
{
    return hipblasCgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  hipblasOperation_t          transB,
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
                                                  int                         ldc)
{
    return hipblasZgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// gemm_batched
template <>
hipblasStatus_t hipblasGemmBatched<hipblasHalf>(hipblasHandle_t          handle,
                                                hipblasOperation_t       transA,
                                                hipblasOperation_t       transB,
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
                                                int                      batch_count)
{
    return hipblasHgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<float>(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
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
                                          int                batch_count)
{
    return hipblasSgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<double>(hipblasHandle_t     handle,
                                           hipblasOperation_t  transA,
                                           hipblasOperation_t  transB,
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
                                           int                 batch_count)
{
    return hipblasDgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   hipblasOperation_t          transA,
                                                   hipblasOperation_t          transB,
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
                                                   int                         batch_count)
{
    return hipblasCgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         hipblasOperation_t                transA,
                                                         hipblasOperation_t                transB,
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
                                                         int batch_count)
{
    return hipblasZgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

// gemm_strided_batched
template <>
hipblasStatus_t hipblasGemmStridedBatched<hipblasHalf>(hipblasHandle_t    handle,
                                                       hipblasOperation_t transA,
                                                       hipblasOperation_t transB,
                                                       int                m,
                                                       int                n,
                                                       int                k,
                                                       const hipblasHalf* alpha,
                                                       const hipblasHalf* A,
                                                       int                lda,
                                                       int                bsa,
                                                       const hipblasHalf* B,
                                                       int                ldb,
                                                       int                bsb,
                                                       const hipblasHalf* beta,
                                                       hipblasHalf*       C,
                                                       int                ldc,
                                                       int                bsc,
                                                       int                batch_count)
{

    return hipblasHgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 hipblasOperation_t transB,
                                                 int                m,
                                                 int                n,
                                                 int                k,
                                                 const float*       alpha,
                                                 const float*       A,
                                                 int                lda,
                                                 int                bsa,
                                                 const float*       B,
                                                 int                ldb,
                                                 int                bsb,
                                                 const float*       beta,
                                                 float*             C,
                                                 int                ldc,
                                                 int                bsc,
                                                 int                batch_count)
{

    return hipblasSgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  hipblasOperation_t transB,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  int                bsa,
                                                  const double*      B,
                                                  int                ldb,
                                                  int                bsb,
                                                  const double*      beta,
                                                  double*            C,
                                                  int                ldc,
                                                  int                bsc,
                                                  int                batch_count)
{

    return hipblasDgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasOperation_t    transA,
                                                          hipblasOperation_t    transB,
                                                          int                   m,
                                                          int                   n,
                                                          int                   k,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   bsa,
                                                          const hipblasComplex* B,
                                                          int                   ldb,
                                                          int                   bsb,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          int                   bsc,
                                                          int                   batch_count)
{

    return hipblasCgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasOperation_t          transA,
                                                                hipblasOperation_t          transB,
                                                                int                         m,
                                                                int                         n,
                                                                int                         k,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int                         bsa,
                                                                const hipblasDoubleComplex* B,
                                                                int                         ldb,
                                                                int                         bsb,
                                                                const hipblasDoubleComplex* beta,
                                                                hipblasDoubleComplex*       C,
                                                                int                         ldc,
                                                                int                         bsc,
                                                                int batch_count)
{

    return hipblasZgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

// trmm
template <>
hipblasStatus_t hipblasTrmm<float>(hipblasHandle_t    handle,
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
                                   int                ldb)
{
    return hipblasStrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrmm<double>(hipblasHandle_t    handle,
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
                                    int                ldb)
{
    return hipblasDtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

// trsm
template <>
hipblasStatus_t hipblasTrsm<float>(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   float*             A,
                                   int                lda,
                                   float*             B,
                                   int                ldb)
{
    return hipblasStrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrsm<double>(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    double*            A,
                                    int                lda,
                                    double*            B,
                                    int                ldb)
{
    return hipblasDtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

// trsm_batched
template <>
hipblasStatus_t hipblasTrsmBatched<float>(hipblasHandle_t    handle,
                                          hipblasSideMode_t  side,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                n,
                                          const float*       alpha,
                                          float* const       A[],
                                          int                lda,
                                          float*             B[],
                                          int                ldb,
                                          int                batch_count)
{
    return hipblasStrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

template <>
hipblasStatus_t hipblasTrsmBatched<double>(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           double* const      A[],
                                           int                lda,
                                           double*            B[],
                                           int                ldb,
                                           int                batch_count)
{
    return hipblasDtrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

// trsm_strided_batched
template <>
hipblasStatus_t hipblasTrsmStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasSideMode_t  side,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                n,
                                                 const float*       alpha,
                                                 float*             A,
                                                 int                lda,
                                                 int                strideA,
                                                 float*             B,
                                                 int                ldb,
                                                 int                strideB,
                                                 int                batch_count)
{
    return hipblasStrsmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasTrsmStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  double*            A,
                                                  int                lda,
                                                  int                strideA,
                                                  double*            B,
                                                  int                ldb,
                                                  int                strideB,
                                                  int                batch_count)
{
    return hipblasDtrsmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      batch_count);
}

// geam
template <>
hipblasStatus_t hipblasGeam<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   const float*       A,
                                   int                lda,
                                   const float*       beta,
                                   const float*       B,
                                   int                ldb,
                                   float*             C,
                                   int                ldc)
{
    return hipblasSgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasGeam<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    hipblasOperation_t transB,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      beta,
                                    const double*      B,
                                    int                ldb,
                                    double*            C,
                                    int                ldc)
{
    return hipblasDgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
