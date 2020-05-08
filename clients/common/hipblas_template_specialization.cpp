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
hipblasStatus_t hipblasGer<float, false>(hipblasHandle_t handle,
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
hipblasStatus_t hipblasGer<double, false>(hipblasHandle_t handle,
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

template <>
hipblasStatus_t hipblasGer<hipblasComplex, false>(hipblasHandle_t       handle,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  const hipblasComplex* y,
                                                  int                   incy,
                                                  hipblasComplex*       A,
                                                  int                   lda)
{

    return hipblasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasGer<hipblasComplex, true>(hipblasHandle_t       handle,
                                                 int                   m,
                                                 int                   n,
                                                 const hipblasComplex* alpha,
                                                 const hipblasComplex* x,
                                                 int                   incx,
                                                 const hipblasComplex* y,
                                                 int                   incy,
                                                 hipblasComplex*       A,
                                                 int                   lda)
{

    return hipblasCgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasGer<hipblasDoubleComplex, false>(hipblasHandle_t             handle,
                                                        int                         m,
                                                        int                         n,
                                                        const hipblasDoubleComplex* alpha,
                                                        const hipblasDoubleComplex* x,
                                                        int                         incx,
                                                        const hipblasDoubleComplex* y,
                                                        int                         incy,
                                                        hipblasDoubleComplex*       A,
                                                        int                         lda)
{

    return hipblasZgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasGer<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                       int                         m,
                                                       int                         n,
                                                       const hipblasDoubleComplex* alpha,
                                                       const hipblasDoubleComplex* x,
                                                       int                         incx,
                                                       const hipblasDoubleComplex* y,
                                                       int                         incy,
                                                       hipblasDoubleComplex*       A,
                                                       int                         lda)
{

    return hipblasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

// ger_batched
template <>
hipblasStatus_t hipblasGerBatched<float, false>(hipblasHandle_t    handle,
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
hipblasStatus_t hipblasGerBatched<double, false>(hipblasHandle_t     handle,
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

template <>
hipblasStatus_t hipblasGerBatched<hipblasComplex, false>(hipblasHandle_t             handle,
                                                         int                         m,
                                                         int                         n,
                                                         const hipblasComplex*       alpha,
                                                         const hipblasComplex* const x[],
                                                         int                         incx,
                                                         const hipblasComplex* const y[],
                                                         int                         incy,
                                                         hipblasComplex* const       A[],
                                                         int                         lda,
                                                         int                         batch_count)
{

    return hipblasCgeruBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

template <>
hipblasStatus_t hipblasGerBatched<hipblasComplex, true>(hipblasHandle_t             handle,
                                                        int                         m,
                                                        int                         n,
                                                        const hipblasComplex*       alpha,
                                                        const hipblasComplex* const x[],
                                                        int                         incx,
                                                        const hipblasComplex* const y[],
                                                        int                         incy,
                                                        hipblasComplex* const       A[],
                                                        int                         lda,
                                                        int                         batch_count)
{

    return hipblasCgercBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

template <>
hipblasStatus_t
    hipblasGerBatched<hipblasDoubleComplex, false>(hipblasHandle_t                   handle,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   hipblasDoubleComplex* const       A[],
                                                   int                               lda,
                                                   int                               batch_count)
{

    return hipblasZgeruBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

template <>
hipblasStatus_t hipblasGerBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                              int                         m,
                                                              int                         n,
                                                              const hipblasDoubleComplex* alpha,
                                                              const hipblasDoubleComplex* const x[],
                                                              int incx,
                                                              const hipblasDoubleComplex* const y[],
                                                              int                         incy,
                                                              hipblasDoubleComplex* const A[],
                                                              int                         lda,
                                                              int batch_count)
{

    return hipblasZgercBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batch_count);
}

// ger_strided_batched
template <>
hipblasStatus_t hipblasGerStridedBatched<float, false>(hipblasHandle_t handle,
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
hipblasStatus_t hipblasGerStridedBatched<double, false>(hipblasHandle_t handle,
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

template <>
hipblasStatus_t hipblasGerStridedBatched<hipblasComplex, false>(hipblasHandle_t       handle,
                                                                int                   m,
                                                                int                   n,
                                                                const hipblasComplex* alpha,
                                                                const hipblasComplex* x,
                                                                int                   incx,
                                                                int                   stridex,
                                                                const hipblasComplex* y,
                                                                int                   incy,
                                                                int                   stridey,
                                                                hipblasComplex*       A,
                                                                int                   lda,
                                                                int                   strideA,
                                                                int                   batch_count)
{

    return hipblasCgeruStridedBatched(
        handle, m, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}

template <>
hipblasStatus_t hipblasGerStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
                                                               int                   m,
                                                               int                   n,
                                                               const hipblasComplex* alpha,
                                                               const hipblasComplex* x,
                                                               int                   incx,
                                                               int                   stridex,
                                                               const hipblasComplex* y,
                                                               int                   incy,
                                                               int                   stridey,
                                                               hipblasComplex*       A,
                                                               int                   lda,
                                                               int                   strideA,
                                                               int                   batch_count)
{

    return hipblasCgercStridedBatched(
        handle, m, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}

template <>
hipblasStatus_t
    hipblasGerStridedBatched<hipblasDoubleComplex, false>(hipblasHandle_t             handle,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          int                         stridey,
                                                          hipblasDoubleComplex*       A,
                                                          int                         lda,
                                                          int                         strideA,
                                                          int                         batch_count)
{

    return hipblasZgeruStridedBatched(
        handle, m, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}

template <>
hipblasStatus_t
    hipblasGerStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                         int                         m,
                                                         int                         n,
                                                         const hipblasDoubleComplex* alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         int                         stridex,
                                                         const hipblasDoubleComplex* y,
                                                         int                         incy,
                                                         int                         stridey,
                                                         hipblasDoubleComplex*       A,
                                                         int                         lda,
                                                         int                         strideA,
                                                         int                         batch_count)
{

    return hipblasZgercStridedBatched(
        handle, m, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batch_count);
}

// hbmv
template <>
hipblasStatus_t hipblasHbmv<hipblasComplex>(hipblasHandle_t       handle,
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
                                            int                   incy)
{
    return hipblasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasHbmv<hipblasDoubleComplex>(hipblasHandle_t             handle,
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
                                                  int                         incy)
{
    return hipblasZhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// hbmv_batched
template <>
hipblasStatus_t hipblasHbmvBatched<hipblasComplex>(hipblasHandle_t             handle,
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
                                                   int                         batchCount)
{
    return hipblasChbmvBatched(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasHbmvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
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
                                                         int batchCount)
{
    return hipblasZhbmvBatched(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

// hbmv_strided_batched
template <>
hipblasStatus_t hipblasHbmvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          int                   k,
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
                                                          int                   batchCount)
{
    return hipblasChbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasHbmvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                int                         n,
                                                                int                         k,
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
                                                                int batchCount)
{
    return hipblasZhbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
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
                                      batchCount);
}

// hemv
template <>
hipblasStatus_t hipblasHemv<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
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
    return hipblasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasHemv<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
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
    return hipblasZhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

// hemv_batched
template <>
hipblasStatus_t hipblasHemvBatched<hipblasComplex>(hipblasHandle_t             handle,
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
                                                   int                         batch_count)
{
    return hipblasChemvBatched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasHemvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
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
                                                         int batch_count)
{
    return hipblasZhemvBatched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

// hemv_strided_batched
template <>
hipblasStatus_t hipblasHemvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
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
    return hipblasChemvStridedBatched(handle,
                                      uplo,
                                      n,
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
hipblasStatus_t hipblasHemvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                int                         n,
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
    return hipblasZhemvStridedBatched(handle,
                                      uplo,
                                      n,
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

// her
template <>
hipblasStatus_t hipblasHer<hipblasComplex, float>(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const float*          alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasComplex*       A,
                                                  int                   lda)
{
    return hipblasCher(handle, uplo, n, alpha, x, incx, A, lda);
}

template <>
hipblasStatus_t hipblasHer<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const double*               alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         hipblasDoubleComplex*       A,
                                                         int                         lda)
{
    return hipblasZher(handle, uplo, n, alpha, x, incx, A, lda);
}

// her_batched
template <>
hipblasStatus_t hipblasHerBatched<hipblasComplex, float>(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const float*                alpha,
                                                         const hipblasComplex* const x[],
                                                         int                         incx,
                                                         hipblasComplex* const       A[],
                                                         int                         lda,
                                                         int                         batchCount)
{
    return hipblasCherBatched(handle, uplo, n, alpha, x, incx, A, lda, batchCount);
}

template <>
hipblasStatus_t
    hipblasHerBatched<hipblasDoubleComplex, double>(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    int                               n,
                                                    const double*                     alpha,
                                                    const hipblasDoubleComplex* const x[],
                                                    int                               incx,
                                                    hipblasDoubleComplex* const       A[],
                                                    int                               lda,
                                                    int                               batchCount)
{
    return hipblasZherBatched(handle, uplo, n, alpha, x, incx, A, lda, batchCount);
}

// her_strided_batched
template <>
hipblasStatus_t hipblasHerStridedBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                                hipblasFillMode_t     uplo,
                                                                int                   n,
                                                                const float*          alpha,
                                                                const hipblasComplex* x,
                                                                int                   incx,
                                                                int                   stridex,
                                                                hipblasComplex*       A,
                                                                int                   lda,
                                                                int                   strideA,
                                                                int                   batchCount)
{
    return hipblasCherStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, A, lda, strideA, batchCount);
}

template <>
hipblasStatus_t
    hipblasHerStridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           int                         n,
                                                           const double*               alpha,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           hipblasDoubleComplex*       A,
                                                           int                         lda,
                                                           int                         strideA,
                                                           int                         batchCount)
{
    return hipblasZherStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, A, lda, strideA, batchCount);
}

// her2
template <>
hipblasStatus_t hipblasHer2<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       A,
                                            int                   lda)
{
    return hipblasCher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasHer2<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  const hipblasDoubleComplex* y,
                                                  int                         incy,
                                                  hipblasDoubleComplex*       A,
                                                  int                         lda)
{
    return hipblasZher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

// her2_batched
template <>
hipblasStatus_t hipblasHer2Batched<hipblasComplex>(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   hipblasComplex* const       A[],
                                                   int                         lda,
                                                   int                         batchCount)
{
    return hipblasCher2Batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batchCount);
}

template <>
hipblasStatus_t hipblasHer2Batched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         hipblasFillMode_t                 uplo,
                                                         int                               n,
                                                         const hipblasDoubleComplex*       alpha,
                                                         const hipblasDoubleComplex* const x[],
                                                         int                               incx,
                                                         const hipblasDoubleComplex* const y[],
                                                         int                               incy,
                                                         hipblasDoubleComplex* const       A[],
                                                         int                               lda,
                                                         int batchCount)
{
    return hipblasZher2Batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batchCount);
}

// her2_strided_batched
template <>
hipblasStatus_t hipblasHer2StridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          int                   strideA,
                                                          int                   batchCount)
{
    return hipblasCher2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batchCount);
}

template <>
hipblasStatus_t hipblasHer2StridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                const hipblasDoubleComplex* y,
                                                                int                         incy,
                                                                int                         stridey,
                                                                hipblasDoubleComplex*       A,
                                                                int                         lda,
                                                                int                         strideA,
                                                                int batchCount)
{
    return hipblasZher2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batchCount);
}

// hpmv
template <>
hipblasStatus_t hipblasHpmv<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* AP,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy)
{
    return hipblasChpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasHpmv<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* AP,
                                                  const hipblasDoubleComplex* x,
                                                  int                         incx,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int                         incy)
{
    return hipblasZhpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

// hpmv_batched
template <>
hipblasStatus_t hipblasHpmvBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const AP[],
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batchCount)
{
    return hipblasChpmvBatched(handle, uplo, n, alpha, AP, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasHpmvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         hipblasFillMode_t                 uplo,
                                                         int                               n,
                                                         const hipblasDoubleComplex*       alpha,
                                                         const hipblasDoubleComplex* const AP[],
                                                         const hipblasDoubleComplex* const x[],
                                                         int                               incx,
                                                         const hipblasDoubleComplex*       beta,
                                                         hipblasDoubleComplex* const       y[],
                                                         int                               incy,
                                                         int batchCount)
{
    return hipblasZhpmvBatched(handle, uplo, n, alpha, AP, x, incx, beta, y, incy, batchCount);
}

// hpmv_strided_batched
template <>
hipblasStatus_t hipblasHpmvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* AP,
                                                          int                   strideAP,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batchCount)
{
    return hipblasChpmvStridedBatched(
        handle, uplo, n, alpha, AP, strideAP, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

template <>
hipblasStatus_t hipblasHpmvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* AP,
                                                                int strideAP,
                                                                const hipblasDoubleComplex* x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                const hipblasDoubleComplex* beta,
                                                                hipblasDoubleComplex*       y,
                                                                int                         incy,
                                                                int                         stridey,
                                                                int batchCount)
{
    return hipblasZhpmvStridedBatched(
        handle, uplo, n, alpha, AP, strideAP, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

// hpr
template <>
hipblasStatus_t hipblasHpr<hipblasComplex, float>(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int                   n,
                                                  const float*          alpha,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasComplex*       AP)
{
    return hipblasChpr(handle, uplo, n, alpha, x, incx, AP);
}

template <>
hipblasStatus_t hipblasHpr<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const double*               alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         hipblasDoubleComplex*       AP)
{
    return hipblasZhpr(handle, uplo, n, alpha, x, incx, AP);
}

// hpr_batched
template <>
hipblasStatus_t hipblasHprBatched<hipblasComplex, float>(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const float*                alpha,
                                                         const hipblasComplex* const x[],
                                                         int                         incx,
                                                         hipblasComplex* const       AP[],
                                                         int                         batchCount)
{
    return hipblasChprBatched(handle, uplo, n, alpha, x, incx, AP, batchCount);
}

template <>
hipblasStatus_t
    hipblasHprBatched<hipblasDoubleComplex, double>(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    int                               n,
                                                    const double*                     alpha,
                                                    const hipblasDoubleComplex* const x[],
                                                    int                               incx,
                                                    hipblasDoubleComplex* const       AP[],
                                                    int                               batchCount)
{
    return hipblasZhprBatched(handle, uplo, n, alpha, x, incx, AP, batchCount);
}

// hpr_strided_batched
template <>
hipblasStatus_t hipblasHprStridedBatched<hipblasComplex, float>(hipblasHandle_t       handle,
                                                                hipblasFillMode_t     uplo,
                                                                int                   n,
                                                                const float*          alpha,
                                                                const hipblasComplex* x,
                                                                int                   incx,
                                                                int                   stridex,
                                                                hipblasComplex*       AP,
                                                                int                   strideAP,
                                                                int                   batchCount)
{
    return hipblasChprStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, AP, strideAP, batchCount);
}

template <>
hipblasStatus_t
    hipblasHprStridedBatched<hipblasDoubleComplex, double>(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           int                         n,
                                                           const double*               alpha,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           hipblasDoubleComplex*       AP,
                                                           int                         strideAP,
                                                           int                         batchCount)
{
    return hipblasZhprStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, AP, strideAP, batchCount);
}

// hpr2
template <>
hipblasStatus_t hipblasHpr2(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            const hipblasComplex* y,
                            int                   incy,
                            hipblasComplex*       AP)
{
    return hipblasChpr2(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

template <>
hipblasStatus_t hipblasHpr2(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            const hipblasDoubleComplex* y,
                            int                         incy,
                            hipblasDoubleComplex*       AP)
{
    return hipblasZhpr2(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

// hpr2_batched
template <>
hipblasStatus_t hipblasHpr2Batched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   const hipblasComplex* const y[],
                                   int                         incy,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount)
{
    return hipblasChpr2Batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batchCount);
}

template <>
hipblasStatus_t hipblasHpr2Batched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   const hipblasDoubleComplex* const y[],
                                   int                               incy,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
{
    return hipblasZhpr2Batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batchCount);
}

// hpr2_strided_batched
template <>
hipblasStatus_t hipblasHpr2StridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          int                   stridex,
                                          const hipblasComplex* y,
                                          int                   incy,
                                          int                   stridey,
                                          hipblasComplex*       AP,
                                          int                   strideAP,
                                          int                   batchCount)
{
    return hipblasChpr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, AP, strideAP, batchCount);
}

template <>
hipblasStatus_t hipblasHpr2StridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          int                         stridex,
                                          const hipblasDoubleComplex* y,
                                          int                         incy,
                                          int                         stridey,
                                          hipblasDoubleComplex*       AP,
                                          int                         strideAP,
                                          int                         batchCount)
{
    return hipblasZhpr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, AP, strideAP, batchCount);
}

// sbmv
template <>
hipblasStatus_t hipblasSbmv(hipblasHandle_t   handle,
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
                            int               incy)
{
    return hipblasSsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasSbmv(hipblasHandle_t   handle,
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
                            int               incy)
{
    return hipblasDsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// sbmv_batched
template <>
hipblasStatus_t hipblasSbmvBatched(hipblasHandle_t    handle,
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
                                   int                batchCount)
{
    return hipblasSsbmvBatched(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasSbmvBatched(hipblasHandle_t     handle,
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
                                   int                 batchCount)
{
    return hipblasDsbmvBatched(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

// sbmv_strided_batched
template <>
hipblasStatus_t hipblasSbmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const float*      alpha,
                                          const float*      A,
                                          int               lda,
                                          int               strideA,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          const float*      beta,
                                          float*            y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount)
{
    return hipblasSsbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSbmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const double*     alpha,
                                          const double*     A,
                                          int               lda,
                                          int               strideA,
                                          const double*     x,
                                          int               incx,
                                          int               stridex,
                                          const double*     beta,
                                          double*           y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount)
{
    return hipblasDsbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
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
                                      batchCount);
}

// spmv
template <>
hipblasStatus_t hipblasSpmv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      AP,
                            const float*      x,
                            int               incx,
                            const float*      beta,
                            float*            y,
                            int               incy)
{
    return hipblasSspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasSpmv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     AP,
                            const double*     x,
                            int               incx,
                            const double*     beta,
                            double*           y,
                            int               incy)
{
    return hipblasDspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

// spmv_batched
template <>
hipblasStatus_t hipblasSpmvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const AP[],
                                   const float* const x[],
                                   int                incx,
                                   const float*       beta,
                                   float*             y[],
                                   int                incy,
                                   int                batchCount)
{
    return hipblasSspmvBatched(handle, uplo, n, alpha, AP, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasSpmvBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const AP[],
                                   const double* const x[],
                                   int                 incx,
                                   const double*       beta,
                                   double*             y[],
                                   int                 incy,
                                   int                 batchCount)
{
    return hipblasDspmvBatched(handle, uplo, n, alpha, AP, x, incx, beta, y, incy, batchCount);
}

// spmv_strided_batched
template <>
hipblasStatus_t hipblasSpmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      AP,
                                          int               strideAP,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          const float*      beta,
                                          float*            y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount)
{
    return hipblasSspmvStridedBatched(
        handle, uplo, n, alpha, AP, strideAP, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

template <>
hipblasStatus_t hipblasSpmvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     AP,
                                          int               strideAP,
                                          const double*     x,
                                          int               incx,
                                          int               stridex,
                                          const double*     beta,
                                          double*           y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount)
{
    return hipblasDspmvStridedBatched(
        handle, uplo, n, alpha, AP, strideAP, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

// spr
template <>
hipblasStatus_t hipblasSpr(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const float*      alpha,
                           const float*      x,
                           int               incx,
                           float*            AP)
{
    return hipblasSspr(handle, uplo, n, alpha, x, incx, AP);
}

template <>
hipblasStatus_t hipblasSpr(hipblasHandle_t   handle,
                           hipblasFillMode_t uplo,
                           int               n,
                           const double*     alpha,
                           const double*     x,
                           int               incx,
                           double*           AP)
{
    return hipblasDspr(handle, uplo, n, alpha, x, incx, AP);
}

template <>
hipblasStatus_t hipblasSpr(hipblasHandle_t       handle,
                           hipblasFillMode_t     uplo,
                           int                   n,
                           const hipblasComplex* alpha,
                           const hipblasComplex* x,
                           int                   incx,
                           hipblasComplex*       AP)
{
    return hipblasCspr(handle, uplo, n, alpha, x, incx, AP);
}

template <>
hipblasStatus_t hipblasSpr(hipblasHandle_t             handle,
                           hipblasFillMode_t           uplo,
                           int                         n,
                           const hipblasDoubleComplex* alpha,
                           const hipblasDoubleComplex* x,
                           int                         incx,
                           hipblasDoubleComplex*       AP)
{
    return hipblasZspr(handle, uplo, n, alpha, x, incx, AP);
}

// spr_batched
template <>
hipblasStatus_t hipblasSprBatched(hipblasHandle_t    handle,
                                  hipblasFillMode_t  uplo,
                                  int                n,
                                  const float*       alpha,
                                  const float* const x[],
                                  int                incx,
                                  float* const       AP[],
                                  int                batchCount)
{
    return hipblasSsprBatched(handle, uplo, n, alpha, x, incx, AP, batchCount);
}

template <>
hipblasStatus_t hipblasSprBatched(hipblasHandle_t     handle,
                                  hipblasFillMode_t   uplo,
                                  int                 n,
                                  const double*       alpha,
                                  const double* const x[],
                                  int                 incx,
                                  double* const       AP[],
                                  int                 batchCount)
{
    return hipblasDsprBatched(handle, uplo, n, alpha, x, incx, AP, batchCount);
}

template <>
hipblasStatus_t hipblasSprBatched(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  int                         n,
                                  const hipblasComplex*       alpha,
                                  const hipblasComplex* const x[],
                                  int                         incx,
                                  hipblasComplex* const       AP[],
                                  int                         batchCount)
{
    return hipblasCsprBatched(handle, uplo, n, alpha, x, incx, AP, batchCount);
}

template <>
hipblasStatus_t hipblasSprBatched(hipblasHandle_t                   handle,
                                  hipblasFillMode_t                 uplo,
                                  int                               n,
                                  const hipblasDoubleComplex*       alpha,
                                  const hipblasDoubleComplex* const x[],
                                  int                               incx,
                                  hipblasDoubleComplex* const       AP[],
                                  int                               batchCount)
{
    return hipblasZsprBatched(handle, uplo, n, alpha, x, incx, AP, batchCount);
}

// spr_strided_batched
template <>
hipblasStatus_t hipblasSprStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const float*      alpha,
                                         const float*      x,
                                         int               incx,
                                         int               stridex,
                                         float*            AP,
                                         int               strideAP,
                                         int               batchCount)
{
    return hipblasSsprStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, AP, strideAP, batchCount);
}

template <>
hipblasStatus_t hipblasSprStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const double*     alpha,
                                         const double*     x,
                                         int               incx,
                                         int               stridex,
                                         double*           AP,
                                         int               strideAP,
                                         int               batchCount)
{
    return hipblasDsprStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, AP, strideAP, batchCount);
}

template <>
hipblasStatus_t hipblasSprStridedBatched(hipblasHandle_t       handle,
                                         hipblasFillMode_t     uplo,
                                         int                   n,
                                         const hipblasComplex* alpha,
                                         const hipblasComplex* x,
                                         int                   incx,
                                         int                   stridex,
                                         hipblasComplex*       AP,
                                         int                   strideAP,
                                         int                   batchCount)
{
    return hipblasCsprStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, AP, strideAP, batchCount);
}

template <>
hipblasStatus_t hipblasSprStridedBatched(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         int                         n,
                                         const hipblasDoubleComplex* alpha,
                                         const hipblasDoubleComplex* x,
                                         int                         incx,
                                         int                         stridex,
                                         hipblasDoubleComplex*       AP,
                                         int                         strideAP,
                                         int                         batchCount)
{
    return hipblasZsprStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, AP, strideAP, batchCount);
}

// spr2
template <>
hipblasStatus_t hipblasSpr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            const float*      y,
                            int               incy,
                            float*            AP)
{
    return hipblasSspr2(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

template <>
hipblasStatus_t hipblasSpr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            const double*     y,
                            int               incy,
                            double*           AP)
{
    return hipblasDspr2(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

// spr2_batched
template <>
hipblasStatus_t hipblasSpr2Batched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   float* const       AP[],
                                   int                batchCount)
{
    return hipblasSspr2Batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batchCount);
}

template <>
hipblasStatus_t hipblasSpr2Batched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   double* const       AP[],
                                   int                 batchCount)
{
    return hipblasDspr2Batched(handle, uplo, n, alpha, x, incx, y, incy, AP, batchCount);
}

// spr2_strided_batched
template <>
hipblasStatus_t hipblasSpr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          const float*      y,
                                          int               incy,
                                          int               stridey,
                                          float*            AP,
                                          int               strideAP,
                                          int               batchCount)
{
    return hipblasSspr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, AP, strideAP, batchCount);
}

template <>
hipblasStatus_t hipblasSpr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          int               stridex,
                                          const double*     y,
                                          int               incy,
                                          int               stridey,
                                          double*           AP,
                                          int               strideAP,
                                          int               batchCount)
{
    return hipblasDspr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, AP, strideAP, batchCount);
}

// symv
template <>
hipblasStatus_t hipblasSymv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      A,
                            int               lda,
                            const float*      x,
                            int               incx,
                            const float*      beta,
                            float*            y,
                            int               incy)
{
    return hipblasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasSymv(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     A,
                            int               lda,
                            const double*     x,
                            int               incx,
                            const double*     beta,
                            double*           y,
                            int               incy)
{
    return hipblasDsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasSymv(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
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
    return hipblasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasSymv(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
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
    return hipblasZsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

// symv_batched
template <>
hipblasStatus_t hipblasSymvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const A[],
                                   int                lda,
                                   const float* const x[],
                                   int                incx,
                                   const float*       beta,
                                   float*             y[],
                                   int                incy,
                                   int                batchCount)
{
    return hipblasSsymvBatched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasSymvBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const A[],
                                   int                 lda,
                                   const double* const x[],
                                   int                 incx,
                                   const double*       beta,
                                   double*             y[],
                                   int                 incy,
                                   int                 batchCount)
{
    return hipblasDsymvBatched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasSymvBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const A[],
                                   int                         lda,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   const hipblasComplex*       beta,
                                   hipblasComplex*             y[],
                                   int                         incy,
                                   int                         batchCount)
{
    return hipblasCsymvBatched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

template <>
hipblasStatus_t hipblasSymvBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const A[],
                                   int                               lda,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   const hipblasDoubleComplex*       beta,
                                   hipblasDoubleComplex*             y[],
                                   int                               incy,
                                   int                               batchCount)
{
    return hipblasZsymvBatched(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batchCount);
}

// symv_strided_batched
template <>
hipblasStatus_t hipblasSymvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      A,
                                          int               lda,
                                          int               strideA,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          const float*      beta,
                                          float*            y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount)
{
    return hipblasSsymvStridedBatched(handle,
                                      uplo,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSymvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     A,
                                          int               lda,
                                          int               strideA,
                                          const double*     x,
                                          int               incx,
                                          int               stridex,
                                          const double*     beta,
                                          double*           y,
                                          int               incy,
                                          int               stridey,
                                          int               batchCount)
{
    return hipblasDsymvStridedBatched(handle,
                                      uplo,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSymvStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
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
                                          int                   batchCount)
{
    return hipblasCsymvStridedBatched(handle,
                                      uplo,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSymvStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
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
                                          int                         batchCount)
{
    return hipblasZsymvStridedBatched(handle,
                                      uplo,
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
                                      batchCount);
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

template <>
hipblasStatus_t hipblasSyr<hipblasComplex>(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasComplex*       A,
                                           int                   lda)
{
    return hipblasCsyr(handle, uplo, n, alpha, x, incx, A, lda);
}

template <>
hipblasStatus_t hipblasSyr<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int                         n,
                                                 const hipblasDoubleComplex* alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int                         incx,
                                                 hipblasDoubleComplex*       A,
                                                 int                         lda)
{
    return hipblasZsyr(handle, uplo, n, alpha, x, incx, A, lda);
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

template <>
hipblasStatus_t hipblasSyrBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasComplex*       alpha,
                                                  const hipblasComplex* const x[],
                                                  int                         incx,
                                                  hipblasComplex* const       A[],
                                                  int                         lda,
                                                  int                         batch_count)
{
    return hipblasCsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count);
}

template <>
hipblasStatus_t hipblasSyrBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                        hipblasFillMode_t                 uplo,
                                                        int                               n,
                                                        const hipblasDoubleComplex*       alpha,
                                                        const hipblasDoubleComplex* const x[],
                                                        int                               incx,
                                                        hipblasDoubleComplex* const       A[],
                                                        int                               lda,
                                                        int batch_count)
{
    return hipblasZsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count);
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

template <>
hipblasStatus_t hipblasSyrStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                         hipblasFillMode_t     uplo,
                                                         int                   n,
                                                         const hipblasComplex* alpha,
                                                         const hipblasComplex* x,
                                                         int                   incx,
                                                         int                   stridex,
                                                         hipblasComplex*       A,
                                                         int                   lda,
                                                         int                   strideA,
                                                         int                   batch_count)
{
    return hipblasCsyrStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, A, lda, strideA, batch_count);
}

template <>
hipblasStatus_t hipblasSyrStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                               hipblasFillMode_t           uplo,
                                                               int                         n,
                                                               const hipblasDoubleComplex* alpha,
                                                               const hipblasDoubleComplex* x,
                                                               int                         incx,
                                                               int                         stridex,
                                                               hipblasDoubleComplex*       A,
                                                               int                         lda,
                                                               int                         strideA,
                                                               int batch_count)
{
    return hipblasZsyrStridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, A, lda, strideA, batch_count);
}

// syr2
template <>
hipblasStatus_t hipblasSyr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            const float*      y,
                            int               incy,
                            float*            A,
                            int               lda)
{
    return hipblasSsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasSyr2(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            const double*     y,
                            int               incy,
                            double*           A,
                            int               lda)
{
    return hipblasDsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasSyr2(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            const hipblasComplex* y,
                            int                   incy,
                            hipblasComplex*       A,
                            int                   lda)
{
    return hipblasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasSyr2(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            const hipblasDoubleComplex* y,
                            int                         incy,
                            hipblasDoubleComplex*       A,
                            int                         lda)
{
    return hipblasZsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

// syr2_batched
template <>
hipblasStatus_t hipblasSyr2Batched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount)
{
    return hipblasSsyr2Batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2Batched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount)
{
    return hipblasDsyr2Batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2Batched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   const hipblasComplex* const y[],
                                   int                         incy,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount)
{
    return hipblasCsyr2Batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2Batched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   const hipblasDoubleComplex* const y[],
                                   int                               incy,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount)
{
    return hipblasZsyr2Batched(handle, uplo, n, alpha, x, incx, y, incy, A, lda, batchCount);
}

// syr2_strided_batched
template <>
hipblasStatus_t hipblasSyr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          const float*      y,
                                          int               incy,
                                          int               stridey,
                                          float*            A,
                                          int               lda,
                                          int               strideA,
                                          int               batchCount)
{
    return hipblasSsyr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2StridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          int               stridex,
                                          const double*     y,
                                          int               incy,
                                          int               stridey,
                                          double*           A,
                                          int               lda,
                                          int               strideA,
                                          int               batchCount)
{
    return hipblasDsyr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2StridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          int                   stridex,
                                          const hipblasComplex* y,
                                          int                   incy,
                                          int                   stridey,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          int                   strideA,
                                          int                   batchCount)
{
    return hipblasCsyr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2StridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          int                         stridex,
                                          const hipblasDoubleComplex* y,
                                          int                         incy,
                                          int                         stridey,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          int                         strideA,
                                          int                         batchCount)
{
    return hipblasZsyr2StridedBatched(
        handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, A, lda, strideA, batchCount);
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

template <>
hipblasStatus_t hipblasTrsv<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx)
{
    return hipblasCtrsv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTrsv<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx)
{
    return hipblasZtrsv(handle, uplo, transA, diag, m, A, lda, x, incx);
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

template <>
hipblasStatus_t hipblasTrsvBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batch_count)
{
    return hipblasCtrsvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTrsvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         hipblasFillMode_t                 uplo,
                                                         hipblasOperation_t                transA,
                                                         hipblasDiagType_t                 diag,
                                                         int                               m,
                                                         const hipblasDoubleComplex* const A[],
                                                         int                               lda,
                                                         hipblasDoubleComplex* const       x[],
                                                         int                               incx,
                                                         int batch_count)
{
    return hipblasZtrsvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
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

template <>
hipblasStatus_t hipblasTrsvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   strideA,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          int                   batch_count)
{
    return hipblasCtrsvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, strideA, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasTrsvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                hipblasOperation_t          transA,
                                                                hipblasDiagType_t           diag,
                                                                int                         m,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int                         strideA,
                                                                hipblasDoubleComplex*       x,
                                                                int                         incx,
                                                                int                         stridex,
                                                                int batch_count)
{
    return hipblasZtrsvStridedBatched(
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

// herk
template <>
hipblasStatus_t hipblasHerk(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            hipblasOperation_t    transA,
                            int                   n,
                            int                   k,
                            const float*          alpha,
                            const hipblasComplex* A,
                            int                   lda,
                            const float*          beta,
                            hipblasComplex*       C,
                            int                   ldc)
{
    return hipblasCherk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasHerk(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            hipblasOperation_t          transA,
                            int                         n,
                            int                         k,
                            const double*               alpha,
                            const hipblasDoubleComplex* A,
                            int                         lda,
                            const double*               beta,
                            hipblasDoubleComplex*       C,
                            int                         ldc)
{
    return hipblasZherk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

// herk_batched
template <>
hipblasStatus_t hipblasHerkBatched(hipblasHandle_t             handle,
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
                                   int                         batchCount)
{
    return hipblasCherkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasHerkBatched(hipblasHandle_t                   handle,
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
                                   int                               batchCount)
{
    return hipblasZherkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

// herk_strided_batched
template <>
hipblasStatus_t hipblasHerkStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          hipblasOperation_t    transA,
                                          int                   n,
                                          int                   k,
                                          const float*          alpha,
                                          const hipblasComplex* A,
                                          int                   lda,
                                          int                   strideA,
                                          const float*          beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          int                   strideC,
                                          int                   batchCount)
{
    return hipblasCherkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

template <>
hipblasStatus_t hipblasHerkStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          hipblasOperation_t          transA,
                                          int                         n,
                                          int                         k,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* A,
                                          int                         lda,
                                          int                         strideA,
                                          const double*               beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          int                         strideC,
                                          int                         batchCount)
{
    return hipblasZherkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

// her2k
template <>
hipblasStatus_t hipblasHer2k(hipblasHandle_t       handle,
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
                             int                   ldc)
{
    return hipblasCher2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasHer2k(hipblasHandle_t             handle,
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
                             int                         ldc)
{
    return hipblasZher2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// her2k_batched
template <>
hipblasStatus_t hipblasHer2kBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return hipblasCher2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasHer2kBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return hipblasZher2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

// her2k_strided_batched
template <>
hipblasStatus_t hipblasHer2kStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
{
    return hipblasCher2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasHer2kStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
{
    return hipblasZher2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// herkx
template <>
hipblasStatus_t hipblasHerkx(hipblasHandle_t       handle,
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
                             int                   ldc)
{
    return hipblasCherkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasHerkx(hipblasHandle_t             handle,
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
                             int                         ldc)
{
    return hipblasZherkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// herkx_batched
template <>
hipblasStatus_t hipblasHerkxBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return hipblasCherkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasHerkxBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return hipblasZherkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

// herkx_strided_batched
template <>
hipblasStatus_t hipblasHerkxStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
{
    return hipblasCherkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasHerkxStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
{
    return hipblasZherkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// symm
template <>
hipblasStatus_t hipblasSymm(hipblasHandle_t   handle,
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
                            int               ldc)
{
    return hipblasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSymm(hipblasHandle_t   handle,
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
                            int               ldc)
{
    return hipblasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSymm(hipblasHandle_t       handle,
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
                            int                   ldc)
{
    return hipblasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSymm(hipblasHandle_t             handle,
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
                            int                         ldc)
{
    return hipblasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

// symm_batched
template <>
hipblasStatus_t hipblasSymmBatched(hipblasHandle_t    handle,
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
                                   int                batchCount)
{
    return hipblasSsymmBatched(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSymmBatched(hipblasHandle_t     handle,
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
                                   int                 batchCount)
{
    return hipblasDsymmBatched(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSymmBatched(hipblasHandle_t             handle,
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
                                   int                         batchCount)
{
    return hipblasCsymmBatched(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSymmBatched(hipblasHandle_t                   handle,
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
                                   int                               batchCount)
{
    return hipblasZsymmBatched(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

// symm_strided_batched
template <>
hipblasStatus_t hipblasSymmStridedBatched(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          hipblasFillMode_t uplo,
                                          int               m,
                                          int               n,
                                          const float*      alpha,
                                          const float*      A,
                                          int               lda,
                                          int               strideA,
                                          const float*      B,
                                          int               ldb,
                                          int               strideB,
                                          const float*      beta,
                                          float*            C,
                                          int               ldc,
                                          int               strideC,
                                          int               batchCount)
{
    return hipblasSsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSymmStridedBatched(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          hipblasFillMode_t uplo,
                                          int               m,
                                          int               n,
                                          const double*     alpha,
                                          const double*     A,
                                          int               lda,
                                          int               strideA,
                                          const double*     B,
                                          int               ldb,
                                          int               strideB,
                                          const double*     beta,
                                          double*           C,
                                          int               ldc,
                                          int               strideC,
                                          int               batchCount)
{
    return hipblasDsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSymmStridedBatched(hipblasHandle_t       handle,
                                          hipblasSideMode_t     side,
                                          hipblasFillMode_t     uplo,
                                          int                   m,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* A,
                                          int                   lda,
                                          int                   strideA,
                                          const hipblasComplex* B,
                                          int                   ldb,
                                          int                   strideB,
                                          const hipblasComplex* beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          int                   strideC,
                                          int                   batchCount)
{
    return hipblasCsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasSymmStridedBatched(hipblasHandle_t             handle,
                                          hipblasSideMode_t           side,
                                          hipblasFillMode_t           uplo,
                                          int                         m,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* A,
                                          int                         lda,
                                          int                         strideA,
                                          const hipblasDoubleComplex* B,
                                          int                         ldb,
                                          int                         strideB,
                                          const hipblasDoubleComplex* beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          int                         strideC,
                                          int                         batchCount)
{
    return hipblasZsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// syrk
template <>
hipblasStatus_t hipblasSyrk(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            int                n,
                            int                k,
                            const float*       alpha,
                            const float*       A,
                            int                lda,
                            const float*       beta,
                            float*             C,
                            int                ldc)
{
    return hipblasSsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrk(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            int                n,
                            int                k,
                            const double*      alpha,
                            const double*      A,
                            int                lda,
                            const double*      beta,
                            double*            C,
                            int                ldc)
{
    return hipblasDsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrk(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            hipblasOperation_t    transA,
                            int                   n,
                            int                   k,
                            const hipblasComplex* alpha,
                            const hipblasComplex* A,
                            int                   lda,
                            const hipblasComplex* beta,
                            hipblasComplex*       C,
                            int                   ldc)
{
    return hipblasCsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrk(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            hipblasOperation_t          transA,
                            int                         n,
                            int                         k,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* A,
                            int                         lda,
                            const hipblasDoubleComplex* beta,
                            hipblasDoubleComplex*       C,
                            int                         ldc)
{
    return hipblasZsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

// syrk_batched
template <>
hipblasStatus_t hipblasSyrkBatched(hipblasHandle_t    handle,
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
                                   int                batchCount)
{
    return hipblasSsyrkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkBatched(hipblasHandle_t     handle,
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
                                   int                 batchCount)
{
    return hipblasDsyrkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkBatched(hipblasHandle_t             handle,
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
                                   int                         batchCount)
{
    return hipblasCsyrkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkBatched(hipblasHandle_t                   handle,
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
                                   int                               batchCount)
{
    return hipblasZsyrkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

// syrk_strided_batched
template <>
hipblasStatus_t hipblasSyrkStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          int                n,
                                          int                k,
                                          const float*       alpha,
                                          const float*       A,
                                          int                lda,
                                          int                strideA,
                                          const float*       beta,
                                          float*             C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount)
{
    return hipblasSsyrkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          int                n,
                                          int                k,
                                          const double*      alpha,
                                          const double*      A,
                                          int                lda,
                                          int                strideA,
                                          const double*      beta,
                                          double*            C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount)
{
    return hipblasDsyrkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          hipblasOperation_t    transA,
                                          int                   n,
                                          int                   k,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* A,
                                          int                   lda,
                                          int                   strideA,
                                          const hipblasComplex* beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          int                   strideC,
                                          int                   batchCount)
{
    return hipblasCsyrkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          hipblasOperation_t          transA,
                                          int                         n,
                                          int                         k,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* A,
                                          int                         lda,
                                          int                         strideA,
                                          const hipblasDoubleComplex* beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          int                         strideC,
                                          int                         batchCount)
{
    return hipblasZsyrkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

// syr2k
template <>
hipblasStatus_t hipblasSyr2k(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipblasSsyr2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyr2k(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipblasDsyr2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyr2k(hipblasHandle_t       handle,
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
                             int                   ldc)
{
    return hipblasCsyr2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyr2k(hipblasHandle_t             handle,
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
                             int                         ldc)
{
    return hipblasZsyr2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// syr2k_batched
template <>
hipblasStatus_t hipblasSyr2kBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    return hipblasSsyr2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    return hipblasDsyr2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return hipblasCsyr2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return hipblasZsyr2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

// syr2k_strided_batched
template <>
hipblasStatus_t hipblasSyr2kStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           int                strideA,
                                           const float*       B,
                                           int                ldb,
                                           int                strideB,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
{
    return hipblasSsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           int                strideA,
                                           const double*      B,
                                           int                ldb,
                                           int                strideB,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
{
    return hipblasDsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
{
    return hipblasCsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
{
    return hipblasZsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// syrkx
template <>
hipblasStatus_t hipblasSyrkx(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipblasSsyrkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrkx(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipblasDsyrkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrkx(hipblasHandle_t       handle,
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
                             int                   ldc)
{
    return hipblasCsyrkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrkx(hipblasHandle_t             handle,
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
                             int                         ldc)
{
    return hipblasZsyrkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// syrkx_batched
template <>
hipblasStatus_t hipblasSyrkxBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    return hipblasSsyrkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    return hipblasDsyrkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return hipblasCsyrkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return hipblasZsyrkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

// syrkx_strided_batched
template <>
hipblasStatus_t hipblasSyrkxStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           int                strideA,
                                           const float*       B,
                                           int                ldb,
                                           int                strideB,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
{
    return hipblasSsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           int                strideA,
                                           const double*      B,
                                           int                ldb,
                                           int                strideB,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
{
    return hipblasDsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
{
    return hipblasCsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
{
    return hipblasZsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// hemm
template <>
hipblasStatus_t hipblasHemm(hipblasHandle_t       handle,
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
                            int                   ldc)
{
    return hipblasChemm(handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasHemm(hipblasHandle_t             handle,
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
                            int                         ldc)
{
    return hipblasZhemm(handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// hemm_batched
template <>
hipblasStatus_t hipblasHemmBatched(hipblasHandle_t             handle,
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
                                   int                         batchCount)
{
    return hipblasChemmBatched(
        handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasHemmBatched(hipblasHandle_t                   handle,
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
                                   int                               batchCount)
{
    return hipblasZhemmBatched(
        handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

// hemm_strided_batched
template <>
hipblasStatus_t hipblasHemmStridedBatched(hipblasHandle_t       handle,
                                          hipblasSideMode_t     side,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          int                   k,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* A,
                                          int                   lda,
                                          int                   strideA,
                                          const hipblasComplex* B,
                                          int                   ldb,
                                          int                   strideB,
                                          const hipblasComplex* beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          int                   strideC,
                                          int                   batchCount)
{
    return hipblasChemmStridedBatched(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasHemmStridedBatched(hipblasHandle_t             handle,
                                          hipblasSideMode_t           side,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          int                         k,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* A,
                                          int                         lda,
                                          int                         strideA,
                                          const hipblasDoubleComplex* B,
                                          int                         ldb,
                                          int                         strideB,
                                          const hipblasDoubleComplex* beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          int                         strideC,
                                          int                         batchCount)
{
    return hipblasZhemmStridedBatched(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
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

template <>
hipblasStatus_t hipblasTrmm<hipblasComplex>(hipblasHandle_t       handle,
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
                                            int                   ldb)
{
    return hipblasCtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrmm<hipblasDoubleComplex>(hipblasHandle_t             handle,
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
                                                  int                         ldb)
{
    return hipblasZtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

// trmm_batched
template <>
hipblasStatus_t hipblasTrmmBatched<float>(hipblasHandle_t    handle,
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
                                          int                batchCount)
{
    return hipblasStrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

template <>
hipblasStatus_t hipblasTrmmBatched<double>(hipblasHandle_t     handle,
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
                                           int                 batchCount)
{
    return hipblasDtrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

template <>
hipblasStatus_t hipblasTrmmBatched<hipblasComplex>(hipblasHandle_t             handle,
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
                                                   int                         batchCount)
{
    return hipblasCtrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

template <>
hipblasStatus_t hipblasTrmmBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
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
                                                         int batchCount)
{
    return hipblasZtrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

// trmm_strided_batched
template <>
hipblasStatus_t hipblasTrmmStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasSideMode_t  side,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                n,
                                                 const float*       alpha,
                                                 const float*       A,
                                                 int                lda,
                                                 int                strideA,
                                                 float*             B,
                                                 int                ldb,
                                                 int                strideB,
                                                 int                batchCount)
{
    return hipblasStrmmStridedBatched(handle,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasTrmmStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasSideMode_t  side,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                n,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  int                strideA,
                                                  double*            B,
                                                  int                ldb,
                                                  int                strideB,
                                                  int                batchCount)
{
    return hipblasDtrmmStridedBatched(handle,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasTrmmStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   strideA,
                                                          hipblasComplex*       B,
                                                          int                   ldb,
                                                          int                   strideB,
                                                          int                   batchCount)
{
    return hipblasCtrmmStridedBatched(handle,
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
                                      batchCount);
}

template <>
hipblasStatus_t hipblasTrmmStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasSideMode_t           side,
                                                                hipblasFillMode_t           uplo,
                                                                hipblasOperation_t          transA,
                                                                hipblasDiagType_t           diag,
                                                                int                         m,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int                         strideA,
                                                                hipblasDoubleComplex*       B,
                                                                int                         ldb,
                                                                int                         strideB,
                                                                int batchCount)
{
    return hipblasZtrmmStridedBatched(handle,
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
                                      batchCount);
}

// tbmv
template <>
hipblasStatus_t hipblasTbmv<float>(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                k,
                                   const float*       A,
                                   int                lda,
                                   float*             x,
                                   int                incx)
{
    return hipblasStbmv(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTbmv<double>(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                k,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx)
{
    return hipblasDtbmv(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTbmv<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            int                   k,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx)
{
    return hipblasCtbmv(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTbmv<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  int                         k,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx)
{
    return hipblasZtbmv(handle, uplo, transA, diag, m, k, A, lda, x, incx);
}

// tbmv_batched
template <>
hipblasStatus_t hipblasTbmvBatched<float>(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                k,
                                          const float* const A[],
                                          int                lda,
                                          float* const       x[],
                                          int                incx,
                                          int                batch_count)
{
    return hipblasStbmvBatched(handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTbmvBatched<double>(hipblasHandle_t     handle,
                                           hipblasFillMode_t   uplo,
                                           hipblasOperation_t  transA,
                                           hipblasDiagType_t   diag,
                                           int                 m,
                                           int                 k,
                                           const double* const A[],
                                           int                 lda,
                                           double* const       x[],
                                           int                 incx,
                                           int                 batch_count)
{
    return hipblasDtbmvBatched(handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTbmvBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   int                         k,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batch_count)
{
    return hipblasCtbmvBatched(handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTbmvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         hipblasFillMode_t                 uplo,
                                                         hipblasOperation_t                transA,
                                                         hipblasDiagType_t                 diag,
                                                         int                               m,
                                                         int                               k,
                                                         const hipblasDoubleComplex* const A[],
                                                         int                               lda,
                                                         hipblasDoubleComplex* const       x[],
                                                         int                               incx,
                                                         int batch_count)
{
    return hipblasZtbmvBatched(handle, uplo, transA, diag, m, k, A, lda, x, incx, batch_count);
}

// tbmv_strided_batched
template <>
hipblasStatus_t hipblasTbmvStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                k,
                                                 const float*       A,
                                                 int                lda,
                                                 int                stride_a,
                                                 float*             x,
                                                 int                incx,
                                                 int                stride_x,
                                                 int                batch_count)
{
    return hipblasStbmvStridedBatched(
        handle, uplo, transA, diag, m, k, A, lda, stride_a, x, incx, stride_x, batch_count);
}

template <>
hipblasStatus_t hipblasTbmvStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  int                k,
                                                  const double*      A,
                                                  int                lda,
                                                  int                stride_a,
                                                  double*            x,
                                                  int                incx,
                                                  int                stride_x,
                                                  int                batch_count)
{
    return hipblasDtbmvStridedBatched(
        handle, uplo, transA, diag, m, k, A, lda, stride_a, x, incx, stride_x, batch_count);
}

template <>
hipblasStatus_t hipblasTbmvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          int                   k,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   stride_a,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          int                   stride_x,
                                                          int                   batch_count)
{
    return hipblasCtbmvStridedBatched(
        handle, uplo, transA, diag, m, k, A, lda, stride_a, x, incx, stride_x, batch_count);
}

template <>
hipblasStatus_t hipblasTbmvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                hipblasOperation_t          transA,
                                                                hipblasDiagType_t           diag,
                                                                int                         m,
                                                                int                         k,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int                   stride_a,
                                                                hipblasDoubleComplex* x,
                                                                int                   incx,
                                                                int                   stride_x,
                                                                int                   batch_count)
{
    return hipblasZtbmvStridedBatched(
        handle, uplo, transA, diag, m, k, A, lda, stride_a, x, incx, stride_x, batch_count);
}

// tbsv
template <>
hipblasStatus_t hipblasTbsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                n,
                            int                k,
                            const float*       A,
                            int                lda,
                            float*             x,
                            int                incx)
{
    return hipblasStbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTbsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                n,
                            int                k,
                            const double*      A,
                            int                lda,
                            double*            x,
                            int                incx)
{
    return hipblasDtbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTbsv(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            hipblasOperation_t    transA,
                            hipblasDiagType_t     diag,
                            int                   n,
                            int                   k,
                            const hipblasComplex* A,
                            int                   lda,
                            hipblasComplex*       x,
                            int                   incx)
{
    return hipblasCtbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTbsv(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            hipblasOperation_t          transA,
                            hipblasDiagType_t           diag,
                            int                         n,
                            int                         k,
                            const hipblasDoubleComplex* A,
                            int                         lda,
                            hipblasDoubleComplex*       x,
                            int                         incx)
{
    return hipblasZtbsv(handle, uplo, transA, diag, n, k, A, lda, x, incx);
}

// tbsv_batched
template <>
hipblasStatus_t hipblasTbsvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                n,
                                   int                k,
                                   const float* const A[],
                                   int                lda,
                                   float* const       x[],
                                   int                incx,
                                   int                batchCount)
{
    return hipblasStbsvBatched(handle, uplo, transA, diag, n, k, A, lda, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTbsvBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   hipblasOperation_t  transA,
                                   hipblasDiagType_t   diag,
                                   int                 n,
                                   int                 k,
                                   const double* const A[],
                                   int                 lda,
                                   double* const       x[],
                                   int                 incx,
                                   int                 batchCount)
{
    return hipblasDtbsvBatched(handle, uplo, transA, diag, n, k, A, lda, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTbsvBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   hipblasOperation_t          transA,
                                   hipblasDiagType_t           diag,
                                   int                         n,
                                   int                         k,
                                   const hipblasComplex* const A[],
                                   int                         lda,
                                   hipblasComplex* const       x[],
                                   int                         incx,
                                   int                         batchCount)
{
    return hipblasCtbsvBatched(handle, uplo, transA, diag, n, k, A, lda, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTbsvBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   hipblasOperation_t                transA,
                                   hipblasDiagType_t                 diag,
                                   int                               n,
                                   int                               k,
                                   const hipblasDoubleComplex* const A[],
                                   int                               lda,
                                   hipblasDoubleComplex* const       x[],
                                   int                               incx,
                                   int                               batchCount)
{
    return hipblasZtbsvBatched(handle, uplo, transA, diag, n, k, A, lda, x, incx, batchCount);
}

// tbsv_strided_batched
template <>
hipblasStatus_t hipblasTbsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                n,
                                          int                k,
                                          const float*       A,
                                          int                lda,
                                          int                strideA,
                                          float*             x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount)
{
    return hipblasStbsvStridedBatched(
        handle, uplo, transA, diag, n, k, A, lda, strideA, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTbsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                n,
                                          int                k,
                                          const double*      A,
                                          int                lda,
                                          int                strideA,
                                          double*            x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount)
{
    return hipblasDtbsvStridedBatched(
        handle, uplo, transA, diag, n, k, A, lda, strideA, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTbsvStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          hipblasOperation_t    transA,
                                          hipblasDiagType_t     diag,
                                          int                   n,
                                          int                   k,
                                          const hipblasComplex* A,
                                          int                   lda,
                                          int                   strideA,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          int                   stridex,
                                          int                   batchCount)
{
    return hipblasCtbsvStridedBatched(
        handle, uplo, transA, diag, n, k, A, lda, strideA, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTbsvStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          hipblasOperation_t          transA,
                                          hipblasDiagType_t           diag,
                                          int                         n,
                                          int                         k,
                                          const hipblasDoubleComplex* A,
                                          int                         lda,
                                          int                         strideA,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          int                         stridex,
                                          int                         batchCount)
{
    return hipblasZtbsvStridedBatched(
        handle, uplo, transA, diag, n, k, A, lda, strideA, x, incx, stridex, batchCount);
}

// tpmv
template <>
hipblasStatus_t hipblasTpmv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const float*       AP,
                            float*             x,
                            int                incx)
{
    return hipblasStpmv(handle, uplo, transA, diag, m, AP, x, incx);
}

template <>
hipblasStatus_t hipblasTpmv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const double*      AP,
                            double*            x,
                            int                incx)
{
    return hipblasDtpmv(handle, uplo, transA, diag, m, AP, x, incx);
}

template <>
hipblasStatus_t hipblasTpmv(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            hipblasOperation_t    transA,
                            hipblasDiagType_t     diag,
                            int                   m,
                            const hipblasComplex* AP,
                            hipblasComplex*       x,
                            int                   incx)
{
    return hipblasCtpmv(handle, uplo, transA, diag, m, AP, x, incx);
}

template <>
hipblasStatus_t hipblasTpmv(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            hipblasOperation_t          transA,
                            hipblasDiagType_t           diag,
                            int                         m,
                            const hipblasDoubleComplex* AP,
                            hipblasDoubleComplex*       x,
                            int                         incx)
{
    return hipblasZtpmv(handle, uplo, transA, diag, m, AP, x, incx);
}

// tpmv_batched
template <>
hipblasStatus_t hipblasTpmvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const float* const AP[],
                                   float* const       x[],
                                   int                incx,
                                   int                batchCount)
{
    return hipblasStpmvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTpmvBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   hipblasOperation_t  transA,
                                   hipblasDiagType_t   diag,
                                   int                 m,
                                   const double* const AP[],
                                   double* const       x[],
                                   int                 incx,
                                   int                 batchCount)
{
    return hipblasDtpmvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTpmvBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   hipblasOperation_t          transA,
                                   hipblasDiagType_t           diag,
                                   int                         m,
                                   const hipblasComplex* const AP[],
                                   hipblasComplex* const       x[],
                                   int                         incx,
                                   int                         batchCount)
{
    return hipblasCtpmvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTpmvBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   hipblasOperation_t                transA,
                                   hipblasDiagType_t                 diag,
                                   int                               m,
                                   const hipblasDoubleComplex* const AP[],
                                   hipblasDoubleComplex* const       x[],
                                   int                               incx,
                                   int                               batchCount)
{
    return hipblasZtpmvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

// tpmv_strided_batched
template <>
hipblasStatus_t hipblasTpmvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const float*       AP,
                                          int                strideAP,
                                          float*             x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount)
{
    return hipblasStpmvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTpmvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const double*      AP,
                                          int                strideAP,
                                          double*            x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount)
{
    return hipblasDtpmvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTpmvStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          hipblasOperation_t    transA,
                                          hipblasDiagType_t     diag,
                                          int                   m,
                                          const hipblasComplex* AP,
                                          int                   strideAP,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          int                   stridex,
                                          int                   batchCount)
{
    return hipblasCtpmvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTpmvStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          hipblasOperation_t          transA,
                                          hipblasDiagType_t           diag,
                                          int                         m,
                                          const hipblasDoubleComplex* AP,
                                          int                         strideAP,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          int                         stridex,
                                          int                         batchCount)
{
    return hipblasZtpmvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

// tpsv
template <>
hipblasStatus_t hipblasTpsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const float*       AP,
                            float*             x,
                            int                incx)
{
    return hipblasStpsv(handle, uplo, transA, diag, m, AP, x, incx);
}

template <>
hipblasStatus_t hipblasTpsv(hipblasHandle_t    handle,
                            hipblasFillMode_t  uplo,
                            hipblasOperation_t transA,
                            hipblasDiagType_t  diag,
                            int                m,
                            const double*      AP,
                            double*            x,
                            int                incx)
{
    return hipblasDtpsv(handle, uplo, transA, diag, m, AP, x, incx);
}

template <>
hipblasStatus_t hipblasTpsv(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            hipblasOperation_t    transA,
                            hipblasDiagType_t     diag,
                            int                   m,
                            const hipblasComplex* AP,
                            hipblasComplex*       x,
                            int                   incx)
{
    return hipblasCtpsv(handle, uplo, transA, diag, m, AP, x, incx);
}

template <>
hipblasStatus_t hipblasTpsv(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            hipblasOperation_t          transA,
                            hipblasDiagType_t           diag,
                            int                         m,
                            const hipblasDoubleComplex* AP,
                            hipblasDoubleComplex*       x,
                            int                         incx)
{
    return hipblasZtpsv(handle, uplo, transA, diag, m, AP, x, incx);
}

// tpsv_batched
template <>
hipblasStatus_t hipblasTpsvBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const float* const AP[],
                                   float* const       x[],
                                   int                incx,
                                   int                batchCount)
{
    return hipblasStpsvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTpsvBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   hipblasOperation_t  transA,
                                   hipblasDiagType_t   diag,
                                   int                 m,
                                   const double* const AP[],
                                   double* const       x[],
                                   int                 incx,
                                   int                 batchCount)
{
    return hipblasDtpsvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTpsvBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   hipblasOperation_t          transA,
                                   hipblasDiagType_t           diag,
                                   int                         m,
                                   const hipblasComplex* const AP[],
                                   hipblasComplex* const       x[],
                                   int                         incx,
                                   int                         batchCount)
{
    return hipblasCtpsvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

template <>
hipblasStatus_t hipblasTpsvBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   hipblasOperation_t                transA,
                                   hipblasDiagType_t                 diag,
                                   int                               m,
                                   const hipblasDoubleComplex* const AP[],
                                   hipblasDoubleComplex* const       x[],
                                   int                               incx,
                                   int                               batchCount)
{
    return hipblasZtpsvBatched(handle, uplo, transA, diag, m, AP, x, incx, batchCount);
}

// tpsv_strided_batched
template <>
hipblasStatus_t hipblasTpsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const float*       AP,
                                          int                strideAP,
                                          float*             x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount)
{
    return hipblasStpsvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTpsvStridedBatched(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const double*      AP,
                                          int                strideAP,
                                          double*            x,
                                          int                incx,
                                          int                stridex,
                                          int                batchCount)
{
    return hipblasDtpsvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTpsvStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          hipblasOperation_t    transA,
                                          hipblasDiagType_t     diag,
                                          int                   m,
                                          const hipblasComplex* AP,
                                          int                   strideAP,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          int                   stridex,
                                          int                   batchCount)
{
    return hipblasCtpsvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

template <>
hipblasStatus_t hipblasTpsvStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          hipblasOperation_t          transA,
                                          hipblasDiagType_t           diag,
                                          int                         m,
                                          const hipblasDoubleComplex* AP,
                                          int                         strideAP,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          int                         stridex,
                                          int                         batchCount)
{
    return hipblasZtpsvStridedBatched(
        handle, uplo, transA, diag, m, AP, strideAP, x, incx, stridex, batchCount);
}

// trmv
template <>
hipblasStatus_t hipblasTrmv<float>(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const float*       A,
                                   int                lda,
                                   float*             x,
                                   int                incx)
{
    return hipblasStrmv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTrmv<double>(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    const double*      A,
                                    int                lda,
                                    double*            x,
                                    int                incx)
{
    return hipblasDtrmv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTrmv<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx)
{
    return hipblasCtrmv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

template <>
hipblasStatus_t hipblasTrmv<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasDoubleComplex*       x,
                                                  int                         incx)
{
    return hipblasZtrmv(handle, uplo, transA, diag, m, A, lda, x, incx);
}

// trmv_batched
template <>
hipblasStatus_t hipblasTrmvBatched<float>(hipblasHandle_t    handle,
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
    return hipblasStrmvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTrmvBatched<double>(hipblasHandle_t     handle,
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
    return hipblasDtrmvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTrmvBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batch_count)
{
    return hipblasCtrmvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasTrmvBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                         hipblasFillMode_t                 uplo,
                                                         hipblasOperation_t                transA,
                                                         hipblasDiagType_t                 diag,
                                                         int                               m,
                                                         const hipblasDoubleComplex* const A[],
                                                         int                               lda,
                                                         hipblasDoubleComplex* const       x[],
                                                         int                               incx,
                                                         int batch_count)
{
    return hipblasZtrmvBatched(handle, uplo, transA, diag, m, A, lda, x, incx, batch_count);
}

// trmv_strided_batched
template <>
hipblasStatus_t hipblasTrmvStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 const float*       A,
                                                 int                lda,
                                                 int                stride_a,
                                                 float*             x,
                                                 int                incx,
                                                 int                stride_x,
                                                 int                batch_count)
{
    return hipblasStrmvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, stride_a, x, incx, stride_x, batch_count);
}

template <>
hipblasStatus_t hipblasTrmvStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  hipblasDiagType_t  diag,
                                                  int                m,
                                                  const double*      A,
                                                  int                lda,
                                                  int                stride_a,
                                                  double*            x,
                                                  int                incx,
                                                  int                stride_x,
                                                  int                batch_count)
{
    return hipblasDtrmvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, stride_a, x, incx, stride_x, batch_count);
}

template <>
hipblasStatus_t hipblasTrmvStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   stride_a,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          int                   stride_x,
                                                          int                   batch_count)
{
    return hipblasCtrmvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, stride_a, x, incx, stride_x, batch_count);
}

template <>
hipblasStatus_t hipblasTrmvStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasFillMode_t           uplo,
                                                                hipblasOperation_t          transA,
                                                                hipblasDiagType_t           diag,
                                                                int                         m,
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                int                   stride_a,
                                                                hipblasDoubleComplex* x,
                                                                int                   incx,
                                                                int                   stride_x,
                                                                int                   batch_count)
{
    return hipblasZtrmvStridedBatched(
        handle, uplo, transA, diag, m, A, lda, stride_a, x, incx, stride_x, batch_count);
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

template <>
hipblasStatus_t hipblasTrsm<hipblasComplex>(hipblasHandle_t       handle,
                                            hipblasSideMode_t     side,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            hipblasComplex*       A,
                                            int                   lda,
                                            hipblasComplex*       B,
                                            int                   ldb)
{
    return hipblasCtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrsm<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int                         m,
                                                  int                         n,
                                                  const hipblasDoubleComplex* alpha,
                                                  hipblasDoubleComplex*       A,
                                                  int                         lda,
                                                  hipblasDoubleComplex*       B,
                                                  int                         ldb)
{
    return hipblasZtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
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

template <>
hipblasStatus_t hipblasTrsmBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                   hipblasSideMode_t     side,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   hipblasDiagType_t     diag,
                                                   int                   m,
                                                   int                   n,
                                                   const hipblasComplex* alpha,
                                                   hipblasComplex* const A[],
                                                   int                   lda,
                                                   hipblasComplex*       B[],
                                                   int                   ldb,
                                                   int                   batch_count)
{
    return hipblasCtrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

template <>
hipblasStatus_t hipblasTrsmBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                         hipblasSideMode_t           side,
                                                         hipblasFillMode_t           uplo,
                                                         hipblasOperation_t          transA,
                                                         hipblasDiagType_t           diag,
                                                         int                         m,
                                                         int                         n,
                                                         const hipblasDoubleComplex* alpha,
                                                         hipblasDoubleComplex* const A[],
                                                         int                         lda,
                                                         hipblasDoubleComplex*       B[],
                                                         int                         ldb,
                                                         int                         batch_count)
{
    return hipblasZtrsmBatched(
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

template <>
hipblasStatus_t hipblasTrsmStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          int                   strideA,
                                                          hipblasComplex*       B,
                                                          int                   ldb,
                                                          int                   strideB,
                                                          int                   batch_count)
{
    return hipblasCtrsmStridedBatched(handle,
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
hipblasStatus_t hipblasTrsmStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                hipblasSideMode_t           side,
                                                                hipblasFillMode_t           uplo,
                                                                hipblasOperation_t          transA,
                                                                hipblasDiagType_t           diag,
                                                                int                         m,
                                                                int                         n,
                                                                const hipblasDoubleComplex* alpha,
                                                                hipblasDoubleComplex*       A,
                                                                int                         lda,
                                                                int                         strideA,
                                                                hipblasDoubleComplex*       B,
                                                                int                         ldb,
                                                                int                         strideB,
                                                                int batch_count)
{
    return hipblasZtrsmStridedBatched(handle,
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

template <>
hipblasStatus_t hipblasGeam(hipblasHandle_t       handle,
                            hipblasOperation_t    transA,
                            hipblasOperation_t    transB,
                            int                   m,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* A,
                            int                   lda,
                            const hipblasComplex* beta,
                            const hipblasComplex* B,
                            int                   ldb,
                            hipblasComplex*       C,
                            int                   ldc)
{
    return hipblasCgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasGeam(hipblasHandle_t             handle,
                            hipblasOperation_t          transA,
                            hipblasOperation_t          transB,
                            int                         m,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* A,
                            int                         lda,
                            const hipblasDoubleComplex* beta,
                            const hipblasDoubleComplex* B,
                            int                         ldb,
                            hipblasDoubleComplex*       C,
                            int                         ldc)
{
    return hipblasZgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

// geam_batched
template <>
hipblasStatus_t hipblasGeamBatched(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
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
                                   int                batchCount)
{
    return hipblasSgeamBatched(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasGeamBatched(hipblasHandle_t     handle,
                                   hipblasOperation_t  transA,
                                   hipblasOperation_t  transB,
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
                                   int                 batchCount)
{
    return hipblasDgeamBatched(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasGeamBatched(hipblasHandle_t             handle,
                                   hipblasOperation_t          transA,
                                   hipblasOperation_t          transB,
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
                                   int                         batchCount)
{
    return hipblasCgeamBatched(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasGeamBatched(hipblasHandle_t                   handle,
                                   hipblasOperation_t                transA,
                                   hipblasOperation_t                transB,
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
                                   int                               batchCount)
{
    return hipblasZgeamBatched(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
}

// geam_strided_batched
template <>
hipblasStatus_t hipblasGeamStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          const float*       alpha,
                                          const float*       A,
                                          int                lda,
                                          int                strideA,
                                          const float*       beta,
                                          const float*       B,
                                          int                ldb,
                                          int                strideB,
                                          float*             C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount)
{
    return hipblasSgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      B,
                                      ldb,
                                      strideB,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasGeamStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          const double*      alpha,
                                          const double*      A,
                                          int                lda,
                                          int                strideA,
                                          const double*      beta,
                                          const double*      B,
                                          int                ldb,
                                          int                strideB,
                                          double*            C,
                                          int                ldc,
                                          int                strideC,
                                          int                batchCount)
{
    return hipblasDgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      B,
                                      ldb,
                                      strideB,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasGeamStridedBatched(hipblasHandle_t       handle,
                                          hipblasOperation_t    transA,
                                          hipblasOperation_t    transB,
                                          int                   m,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* A,
                                          int                   lda,
                                          int                   strideA,
                                          const hipblasComplex* beta,
                                          const hipblasComplex* B,
                                          int                   ldb,
                                          int                   strideB,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          int                   strideC,
                                          int                   batchCount)
{
    return hipblasCgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      B,
                                      ldb,
                                      strideB,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

template <>
hipblasStatus_t hipblasGeamStridedBatched(hipblasHandle_t             handle,
                                          hipblasOperation_t          transA,
                                          hipblasOperation_t          transB,
                                          int                         m,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* A,
                                          int                         lda,
                                          int                         strideA,
                                          const hipblasDoubleComplex* beta,
                                          const hipblasDoubleComplex* B,
                                          int                         ldb,
                                          int                         strideB,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          int                         strideC,
                                          int                         batchCount)
{
    return hipblasZgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      B,
                                      ldb,
                                      strideB,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

#ifdef __HIP_PLATFORM_SOLVER__

// getrf
template <>
hipblasStatus_t hipblasGetrf<float>(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info)
{
    return hipblasSgetrf(handle, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGetrf<double>(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info)
{
    return hipblasDgetrf(handle, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGetrf<hipblasComplex>(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info)
{
    return hipblasCgetrf(handle, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGetrf<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   int*                  ipiv,
                                                   int*                  info)
{
    return hipblasZgetrf(handle, n, A, lda, ipiv, info);
}

// getrf_batched
template <>
hipblasStatus_t hipblasGetrfBatched<float>(hipblasHandle_t handle,
                                           const int       n,
                                           float* const    A[],
                                           const int       lda,
                                           int*            ipiv,
                                           int*            info,
                                           const int       batchCount)
{
    return hipblasSgetrfBatched(handle, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfBatched<double>(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batchCount)
{
    return hipblasDgetrfBatched(handle, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                    const int             n,
                                                    hipblasComplex* const A[],
                                                    const int             lda,
                                                    int*                  ipiv,
                                                    int*                  info,
                                                    const int             batchCount)
{
    return hipblasCgetrfBatched(handle, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                          const int                   n,
                                                          hipblasDoubleComplex* const A[],
                                                          const int                   lda,
                                                          int*                        ipiv,
                                                          int*                        info,
                                                          const int                   batchCount)
{
    return hipblasZgetrfBatched(handle, n, A, lda, ipiv, info, batchCount);
}

// getrf_strided_batched
template <>
hipblasStatus_t hipblasGetrfStridedBatched<float>(hipblasHandle_t handle,
                                                  const int       n,
                                                  float*          A,
                                                  const int       lda,
                                                  const int       strideA,
                                                  int*            ipiv,
                                                  const int       strideP,
                                                  int*            info,
                                                  const int       batchCount)
{
    return hipblasSgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<double>(hipblasHandle_t handle,
                                                   const int       n,
                                                   double*         A,
                                                   const int       lda,
                                                   const int       strideA,
                                                   int*            ipiv,
                                                   const int       strideP,
                                                   int*            info,
                                                   const int       batchCount)
{
    return hipblasDgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<hipblasComplex>(hipblasHandle_t handle,
                                                           const int       n,
                                                           hipblasComplex* A,
                                                           const int       lda,
                                                           const int       strideA,
                                                           int*            ipiv,
                                                           const int       strideP,
                                                           int*            info,
                                                           const int       batchCount)
{
    return hipblasCgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                                 const int             n,
                                                                 hipblasDoubleComplex* A,
                                                                 const int             lda,
                                                                 const int             strideA,
                                                                 int*                  ipiv,
                                                                 const int             strideP,
                                                                 int*                  info,
                                                                 const int             batchCount)
{
    return hipblasZgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

// getrs
template <>
hipblasStatus_t hipblasGetrs<float>(hipblasHandle_t          handle,
                                    const hipblasOperation_t trans,
                                    const int                n,
                                    const int                nrhs,
                                    float*                   A,
                                    const int                lda,
                                    const int*               ipiv,
                                    float*                   B,
                                    const int                ldb,
                                    int*                     info)
{
    return hipblasSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

template <>
hipblasStatus_t hipblasGetrs<double>(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     double*                  A,
                                     const int                lda,
                                     const int*               ipiv,
                                     double*                  B,
                                     const int                ldb,
                                     int*                     info)
{
    return hipblasDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

template <>
hipblasStatus_t hipblasGetrs<hipblasComplex>(hipblasHandle_t          handle,
                                             const hipblasOperation_t trans,
                                             const int                n,
                                             const int                nrhs,
                                             hipblasComplex*          A,
                                             const int                lda,
                                             const int*               ipiv,
                                             hipblasComplex*          B,
                                             const int                ldb,
                                             int*                     info)
{
    return hipblasCgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

template <>
hipblasStatus_t hipblasGetrs<hipblasDoubleComplex>(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipblasDoubleComplex*    A,
                                                   const int                lda,
                                                   const int*               ipiv,
                                                   hipblasDoubleComplex*    B,
                                                   const int                ldb,
                                                   int*                     info)
{
    return hipblasZgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

// getrs_batched
template <>
hipblasStatus_t hipblasGetrsBatched<float>(hipblasHandle_t          handle,
                                           const hipblasOperation_t trans,
                                           const int                n,
                                           const int                nrhs,
                                           float* const             A[],
                                           const int                lda,
                                           const int*               ipiv,
                                           float* const             B[],
                                           const int                ldb,
                                           int*                     info,
                                           const int                batchCount)
{
    return hipblasSgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsBatched<double>(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            double* const            A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            double* const            B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batchCount)
{
    return hipblasDgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsBatched<hipblasComplex>(hipblasHandle_t          handle,
                                                    const hipblasOperation_t trans,
                                                    const int                n,
                                                    const int                nrhs,
                                                    hipblasComplex* const    A[],
                                                    const int                lda,
                                                    const int*               ipiv,
                                                    hipblasComplex* const    B[],
                                                    const int                ldb,
                                                    int*                     info,
                                                    const int                batchCount)
{
    return hipblasCgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                          const hipblasOperation_t    trans,
                                                          const int                   n,
                                                          const int                   nrhs,
                                                          hipblasDoubleComplex* const A[],
                                                          const int                   lda,
                                                          const int*                  ipiv,
                                                          hipblasDoubleComplex* const B[],
                                                          const int                   ldb,
                                                          int*                        info,
                                                          const int                   batchCount)
{
    return hipblasZgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

// getrs_strided_batched
template <>
hipblasStatus_t hipblasGetrsStridedBatched<float>(hipblasHandle_t          handle,
                                                  const hipblasOperation_t trans,
                                                  const int                n,
                                                  const int                nrhs,
                                                  float*                   A,
                                                  const int                lda,
                                                  const int                strideA,
                                                  const int*               ipiv,
                                                  const int                strideP,
                                                  float*                   B,
                                                  const int                ldb,
                                                  const int                strideB,
                                                  int*                     info,
                                                  const int                batchCount)
{
    return hipblasSgetrsStridedBatched(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsStridedBatched<double>(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   double*                  A,
                                                   const int                lda,
                                                   const int                strideA,
                                                   const int*               ipiv,
                                                   const int                strideP,
                                                   double*                  B,
                                                   const int                ldb,
                                                   const int                strideB,
                                                   int*                     info,
                                                   const int                batchCount)
{
    return hipblasDgetrsStridedBatched(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsStridedBatched<hipblasComplex>(hipblasHandle_t          handle,
                                                           const hipblasOperation_t trans,
                                                           const int                n,
                                                           const int                nrhs,
                                                           hipblasComplex*          A,
                                                           const int                lda,
                                                           const int                strideA,
                                                           const int*               ipiv,
                                                           const int                strideP,
                                                           hipblasComplex*          B,
                                                           const int                ldb,
                                                           const int                strideB,
                                                           int*                     info,
                                                           const int                batchCount)
{
    return hipblasCgetrsStridedBatched(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsStridedBatched<hipblasDoubleComplex>(hipblasHandle_t          handle,
                                                                 const hipblasOperation_t trans,
                                                                 const int                n,
                                                                 const int                nrhs,
                                                                 hipblasDoubleComplex*    A,
                                                                 const int                lda,
                                                                 const int                strideA,
                                                                 const int*               ipiv,
                                                                 const int                strideP,
                                                                 hipblasDoubleComplex*    B,
                                                                 const int                ldb,
                                                                 const int                strideB,
                                                                 int*                     info,
                                                                 const int batchCount)
{
    return hipblasZgetrsStridedBatched(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
}

// geqrf
template <>
hipblasStatus_t hipblasGeqrf<float>(hipblasHandle_t handle,
                                    const int       m,
                                    const int       n,
                                    float*          A,
                                    const int       lda,
                                    float*          ipiv,
                                    int*            info)
{
    return hipblasSgeqrf(handle, m, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGeqrf<double>(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     double*         A,
                                     const int       lda,
                                     double*         ipiv,
                                     int*            info)
{
    return hipblasDgeqrf(handle, m, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGeqrf<hipblasComplex>(hipblasHandle_t handle,
                                             const int       m,
                                             const int       n,
                                             hipblasComplex* A,
                                             const int       lda,
                                             hipblasComplex* ipiv,
                                             int*            info)
{
    return hipblasCgeqrf(handle, m, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGeqrf<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                   const int             m,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   hipblasDoubleComplex* ipiv,
                                                   int*                  info)
{
    return hipblasZgeqrf(handle, m, n, A, lda, ipiv, info);
}

// geqrf_batched
template <>
hipblasStatus_t hipblasGeqrfBatched<float>(hipblasHandle_t handle,
                                           const int       m,
                                           const int       n,
                                           float* const    A[],
                                           const int       lda,
                                           float* const    ipiv[],
                                           int*            info,
                                           const int       batchCount)
{
    return hipblasSgeqrfBatched(handle, m, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfBatched<double>(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            double* const   ipiv[],
                                            int*            info,
                                            const int       batchCount)
{
    return hipblasDgeqrfBatched(handle, m, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                    const int             m,
                                                    const int             n,
                                                    hipblasComplex* const A[],
                                                    const int             lda,
                                                    hipblasComplex* const ipiv[],
                                                    int*                  info,
                                                    const int             batchCount)
{
    return hipblasCgeqrfBatched(handle, m, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                          const int                   m,
                                                          const int                   n,
                                                          hipblasDoubleComplex* const A[],
                                                          const int                   lda,
                                                          hipblasDoubleComplex* const ipiv[],
                                                          int*                        info,
                                                          const int                   batchCount)
{
    return hipblasZgeqrfBatched(handle, m, n, A, lda, ipiv, info, batchCount);
}

// geqrf_strided_batched
template <>
hipblasStatus_t hipblasGeqrfStridedBatched<float>(hipblasHandle_t handle,
                                                  const int       m,
                                                  const int       n,
                                                  float*          A,
                                                  const int       lda,
                                                  const int       strideA,
                                                  float*          ipiv,
                                                  const int       strideP,
                                                  int*            info,
                                                  const int       batchCount)
{
    return hipblasSgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<double>(hipblasHandle_t handle,
                                                   const int       m,
                                                   const int       n,
                                                   double*         A,
                                                   const int       lda,
                                                   const int       strideA,
                                                   double*         ipiv,
                                                   const int       strideP,
                                                   int*            info,
                                                   const int       batchCount)
{
    return hipblasDgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<hipblasComplex>(hipblasHandle_t handle,
                                                           const int       m,
                                                           const int       n,
                                                           hipblasComplex* A,
                                                           const int       lda,
                                                           const int       strideA,
                                                           hipblasComplex* ipiv,
                                                           const int       strideP,
                                                           int*            info,
                                                           const int       batchCount)
{
    return hipblasCgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                                 const int             m,
                                                                 const int             n,
                                                                 hipblasDoubleComplex* A,
                                                                 const int             lda,
                                                                 const int             strideA,
                                                                 hipblasDoubleComplex* ipiv,
                                                                 const int             strideP,
                                                                 int*                  info,
                                                                 const int             batchCount)
{
    return hipblasZgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

#endif
