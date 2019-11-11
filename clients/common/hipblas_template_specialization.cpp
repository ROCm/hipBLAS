/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
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
hipblasStatus_t hipblasAxpy<hipComplex>(hipblasHandle_t   handle,
                                        int               n,
                                        const hipComplex* alpha,
                                        const hipComplex* x,
                                        int               incx,
                                        hipComplex*       y,
                                        int               incy)
{
    return hipblasCaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasAxpy<hipDoubleComplex>(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipDoubleComplex*       y,
                                              int                     incy)
{
    return hipblasZaxpy(handle, n, alpha, x, incx, y, incy);
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
hipblasStatus_t hipblasScal<hipComplex>(
    hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx)
{
    return hipblasCscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipComplex, float>(
    hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx)
{
    return hipblasCsscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipDoubleComplex>(
    hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx)
{
    return hipblasZscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasScal<hipDoubleComplex, double>(
    hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx)
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
hipblasStatus_t hipblasScalBatched<hipComplex>(hipblasHandle_t   handle,
                                               int               n,
                                               const hipComplex* alpha,
                                               hipComplex* const x[],
                                               int               incx,
                                               int               batch_count)
{
    return hipblasCscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipDoubleComplex>(hipblasHandle_t         handle,
                                                     int                     n,
                                                     const hipDoubleComplex* alpha,
                                                     hipDoubleComplex* const x[],
                                                     int                     incx,
                                                     int                     batch_count)
{
    return hipblasZscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipComplex, float>(hipblasHandle_t   handle,
                                                      int               n,
                                                      const float*      alpha,
                                                      hipComplex* const x[],
                                                      int               incx,
                                                      int               batch_count)
{
    return hipblasCsscalBatched(handle, n, alpha, x, incx, batch_count);
}

template <>
hipblasStatus_t hipblasScalBatched<hipDoubleComplex, double>(hipblasHandle_t         handle,
                                                             int                     n,
                                                             const double*           alpha,
                                                             hipDoubleComplex* const x[],
                                                             int                     incx,
                                                             int                     batch_count)
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
hipblasStatus_t hipblasScalStridedBatched<hipComplex>(hipblasHandle_t   handle,
                                                      int               n,
                                                      const hipComplex* alpha,
                                                      hipComplex*       x,
                                                      int               incx,
                                                      int               stridex,
                                                      int               batch_count)
{
    return hipblasCscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipDoubleComplex>(hipblasHandle_t         handle,
                                                            int                     n,
                                                            const hipDoubleComplex* alpha,
                                                            hipDoubleComplex*       x,
                                                            int                     incx,
                                                            int                     stridex,
                                                            int                     batch_count)
{
    return hipblasZscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipComplex, float>(hipblasHandle_t handle,
                                                             int             n,
                                                             const float*    alpha,
                                                             hipComplex*     x,
                                                             int             incx,
                                                             int             stridex,
                                                             int             batch_count)
{
    return hipblasCsscalStridedBatched(handle, n, alpha, x, incx, stridex, batch_count);
}

template <>
hipblasStatus_t hipblasScalStridedBatched<hipDoubleComplex, double>(hipblasHandle_t   handle,
                                                                    int               n,
                                                                    const double*     alpha,
                                                                    hipDoubleComplex* x,
                                                                    int               incx,
                                                                    int               stridex,
                                                                    int               batch_count)
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
hipblasStatus_t hipblasSwap<hipComplex>(
    hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy)
{
    return hipblasCswap(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasSwap<hipDoubleComplex>(
    hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
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
hipblasStatus_t hipblasSwapBatched<hipComplex>(hipblasHandle_t handle,
                                               int             n,
                                               hipComplex*     x[],
                                               int             incx,
                                               hipComplex*     y[],
                                               int             incy,
                                               int             batch_count)
{
    return hipblasCswapBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasSwapBatched<hipDoubleComplex>(hipblasHandle_t   handle,
                                                     int               n,
                                                     hipDoubleComplex* x[],
                                                     int               incx,
                                                     hipDoubleComplex* y[],
                                                     int               incy,
                                                     int               batch_count)
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
hipblasStatus_t hipblasSwapStridedBatched<hipComplex>(hipblasHandle_t handle,
                                                      int             n,
                                                      hipComplex*     x,
                                                      int             incx,
                                                      int             stridex,
                                                      hipComplex*     y,
                                                      int             incy,
                                                      int             stridey,
                                                      int             batch_count)
{
    return hipblasCswapStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasSwapStridedBatched<hipDoubleComplex>(hipblasHandle_t   handle,
                                                            int               n,
                                                            hipDoubleComplex* x,
                                                            int               incx,
                                                            int               stridex,
                                                            hipDoubleComplex* y,
                                                            int               incy,
                                                            int               stridey,
                                                            int               batch_count)
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
hipblasStatus_t hipblasCopy<hipComplex>(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, hipComplex* y, int incy)
{
    return hipblasCcopy(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasCopy<hipDoubleComplex>(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipDoubleComplex*       y,
                                              int                     incy)
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
hipblasStatus_t hipblasCopyBatched<hipComplex>(hipblasHandle_t         handle,
                                               int                     n,
                                               const hipComplex* const x[],
                                               int                     incx,
                                               hipComplex* const       y[],
                                               int                     incy,
                                               int                     batch_count)
{
    return hipblasCcopyBatched(handle, n, x, incx, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasCopyBatched<hipDoubleComplex>(hipblasHandle_t               handle,
                                                     int                           n,
                                                     const hipDoubleComplex* const x[],
                                                     int                           incx,
                                                     hipDoubleComplex* const       y[],
                                                     int                           incy,
                                                     int                           batch_count)
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
hipblasStatus_t hipblasCopyStridedBatched<hipComplex>(hipblasHandle_t   handle,
                                                      int               n,
                                                      const hipComplex* x,
                                                      int               incx,
                                                      int               stridex,
                                                      hipComplex*       y,
                                                      int               incy,
                                                      int               stridey,
                                                      int               batch_count)
{
    return hipblasCcopyStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

template <>
hipblasStatus_t hipblasCopyStridedBatched<hipDoubleComplex>(hipblasHandle_t         handle,
                                                            int                     n,
                                                            const hipDoubleComplex* x,
                                                            int                     incx,
                                                            int                     stridex,
                                                            hipDoubleComplex*       y,
                                                            int                     incy,
                                                            int                     stridey,
                                                            int                     batch_count)
{
    return hipblasZcopyStridedBatched(handle, n, x, incx, stridex, y, incy, stridey, batch_count);
}

// dot
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
hipblasStatus_t hipblasDot<hipComplex>(hipblasHandle_t   handle,
                                       int               n,
                                       const hipComplex* x,
                                       int               incx,
                                       const hipComplex* y,
                                       int               incy,
                                       hipComplex*       result)
{
    return hipblasCdotu(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<hipDoubleComplex>(hipblasHandle_t         handle,
                                             int                     n,
                                             const hipDoubleComplex* x,
                                             int                     incx,
                                             const hipDoubleComplex* y,
                                             int                     incy,
                                             hipDoubleComplex*       result)
{
    return hipblasZdotu(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDotc<hipComplex>(hipblasHandle_t   handle,
                                        int               n,
                                        const hipComplex* x,
                                        int               incx,
                                        const hipComplex* y,
                                        int               incy,
                                        hipComplex*       result)
{
    return hipblasCdotc(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDotc<hipDoubleComplex>(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipDoubleComplex*       result)
{
    return hipblasZdotc(handle, n, x, incx, y, incy, result);
}

// dot_batched
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
hipblasStatus_t hipblasDotBatched<hipComplex>(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipComplex* const x[],
                                              int                     incx,
                                              const hipComplex* const y[],
                                              int                     incy,
                                              int                     batch_count,
                                              hipComplex*             result)
{
    return hipblasCdotuBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcBatched<hipComplex>(hipblasHandle_t         handle,
                                               int                     n,
                                               const hipComplex* const x[],
                                               int                     incx,
                                               const hipComplex* const y[],
                                               int                     incy,
                                               int                     batch_count,
                                               hipComplex*             result)
{
    return hipblasCdotcBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotBatched<hipDoubleComplex>(hipblasHandle_t               handle,
                                                    int                           n,
                                                    const hipDoubleComplex* const x[],
                                                    int                           incx,
                                                    const hipDoubleComplex* const y[],
                                                    int                           incy,
                                                    int                           batch_count,
                                                    hipDoubleComplex*             result)
{
    return hipblasZdotuBatched(handle, n, x, incx, y, incy, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcBatched<hipDoubleComplex>(hipblasHandle_t               handle,
                                                     int                           n,
                                                     const hipDoubleComplex* const x[],
                                                     int                           incx,
                                                     const hipDoubleComplex* const y[],
                                                     int                           incy,
                                                     int                           batch_count,
                                                     hipDoubleComplex*             result)
{
    return hipblasZdotcBatched(handle, n, x, incx, y, incy, batch_count, result);
}

// dot_strided_batched
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
hipblasStatus_t hipblasDotStridedBatched<hipComplex>(hipblasHandle_t   handle,
                                                     int               n,
                                                     const hipComplex* x,
                                                     int               incx,
                                                     int               stridex,
                                                     const hipComplex* y,
                                                     int               incy,
                                                     int               stridey,
                                                     int               batch_count,
                                                     hipComplex*       result)
{
    return hipblasCdotuStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcStridedBatched<hipComplex>(hipblasHandle_t   handle,
                                                      int               n,
                                                      const hipComplex* x,
                                                      int               incx,
                                                      int               stridex,
                                                      const hipComplex* y,
                                                      int               incy,
                                                      int               stridey,
                                                      int               batch_count,
                                                      hipComplex*       result)
{
    return hipblasCdotcStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotStridedBatched<hipDoubleComplex>(hipblasHandle_t         handle,
                                                           int                     n,
                                                           const hipDoubleComplex* x,
                                                           int                     incx,
                                                           int                     stridex,
                                                           const hipDoubleComplex* y,
                                                           int                     incy,
                                                           int                     stridey,
                                                           int                     batch_count,
                                                           hipDoubleComplex*       result)
{
    return hipblasZdotuStridedBatched(
        handle, n, x, incx, stridex, y, incy, stridey, batch_count, result);
}

template <>
hipblasStatus_t hipblasDotcStridedBatched<hipDoubleComplex>(hipblasHandle_t         handle,
                                                            int                     n,
                                                            const hipDoubleComplex* x,
                                                            int                     incx,
                                                            int                     stridex,
                                                            const hipDoubleComplex* y,
                                                            int                     incy,
                                                            int                     stridey,
                                                            int                     batch_count,
                                                            hipDoubleComplex*       result)
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
hipblasStatus_t hipblasAsum<hipComplex, float>(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
{

    return hipblasScasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasAsum<hipDoubleComplex, double>(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
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
hipblasStatus_t hipblasAsumBatched<hipComplex, float>(hipblasHandle_t         handle,
                                                      int                     n,
                                                      const hipComplex* const x[],
                                                      int                     incx,
                                                      int                     batch_count,
                                                      float*                  result)
{

    return hipblasScasumBatched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasAsumBatched<hipDoubleComplex, double>(hipblasHandle_t               handle,
                                                             int                           n,
                                                             const hipDoubleComplex* const x[],
                                                             int                           incx,
                                                             int     batch_count,
                                                             double* result)
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
hipblasStatus_t hipblasAsumStridedBatched<hipComplex, float>(hipblasHandle_t   handle,
                                                             int               n,
                                                             const hipComplex* x,
                                                             int               incx,
                                                             int               stridex,
                                                             int               batch_count,
                                                             float*            result)
{

    return hipblasScasumStridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasAsumStridedBatched<hipDoubleComplex, double>(hipblasHandle_t         handle,
                                                                    int                     n,
                                                                    const hipDoubleComplex* x,
                                                                    int                     incx,
                                                                    int                     stridex,
                                                                    int     batch_count,
                                                                    double* result)
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
hipblasStatus_t hipblasNrm2<hipComplex, float>(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
{

    return hipblasScnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasNrm2<hipDoubleComplex, double>(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
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
hipblasStatus_t hipblasNrm2Batched<hipComplex, float>(hipblasHandle_t         handle,
                                                      int                     n,
                                                      const hipComplex* const x[],
                                                      int                     incx,
                                                      int                     batch_count,
                                                      float*                  result)
{

    return hipblasScnrm2Batched(handle, n, x, incx, batch_count, result);
}

template <>
hipblasStatus_t hipblasNrm2Batched<hipDoubleComplex, double>(hipblasHandle_t               handle,
                                                             int                           n,
                                                             const hipDoubleComplex* const x[],
                                                             int                           incx,
                                                             int     batch_count,
                                                             double* result)
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
hipblasStatus_t hipblasNrm2StridedBatched<hipComplex, float>(hipblasHandle_t   handle,
                                                             int               n,
                                                             const hipComplex* x,
                                                             int               incx,
                                                             int               stridex,
                                                             int               batch_count,
                                                             float*            result)
{

    return hipblasScnrm2StridedBatched(handle, n, x, incx, stridex, batch_count, result);
}

template <>
hipblasStatus_t hipblasNrm2StridedBatched<hipDoubleComplex, double>(hipblasHandle_t         handle,
                                                                    int                     n,
                                                                    const hipDoubleComplex* x,
                                                                    int                     incx,
                                                                    int                     stridex,
                                                                    int     batch_count,
                                                                    double* result)
{

    return hipblasDznrm2StridedBatched(handle, n, x, incx, stridex, batch_count, result);
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
hipblasStatus_t hipblasIamax<hipComplex>(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
{
    return hipblasIcamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasIamax<hipDoubleComplex>(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamax(handle, n, x, incx, result);
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
hipblasStatus_t hipblasIamin<hipComplex>(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
{
    return hipblasIcamin(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasIamin<hipDoubleComplex>(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamin(handle, n, x, incx, result);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

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
hipblasStatus_t hipblasGemv<hipComplex>(hipblasHandle_t    handle,
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
                                        int                incy)
{
    return hipblasCgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGemv<hipDoubleComplex>(hipblasHandle_t         handle,
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
                                              int                     incy)
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
hipblasStatus_t hipblasGemvBatched<hipComplex>(hipblasHandle_t         handle,
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
                                               int                     batch_count)
{
    return hipblasCgemvBatched(
        handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count);
}

template <>
hipblasStatus_t hipblasGemvBatched<hipDoubleComplex>(hipblasHandle_t               handle,
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
                                                     int                           batch_count)
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
hipblasStatus_t hipblasGemvStridedBatched<hipComplex>(hipblasHandle_t    handle,
                                                      hipblasOperation_t transA,
                                                      int                m,
                                                      int                n,
                                                      const hipComplex*  alpha,
                                                      const hipComplex*  A,
                                                      int                lda,
                                                      int                strideA,
                                                      const hipComplex*  x,
                                                      int                incx,
                                                      int                stridex,
                                                      const hipComplex*  beta,
                                                      hipComplex*        y,
                                                      int                incy,
                                                      int                stridey,
                                                      int                batch_count)
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
hipblasStatus_t hipblasGemvStridedBatched<hipDoubleComplex>(hipblasHandle_t         handle,
                                                            hipblasOperation_t      transA,
                                                            int                     m,
                                                            int                     n,
                                                            const hipDoubleComplex* alpha,
                                                            const hipDoubleComplex* A,
                                                            int                     lda,
                                                            int                     strideA,
                                                            const hipDoubleComplex* x,
                                                            int                     incx,
                                                            int                     stridex,
                                                            const hipDoubleComplex* beta,
                                                            hipDoubleComplex*       y,
                                                            int                     incy,
                                                            int                     stridey,
                                                            int                     batch_count)
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
