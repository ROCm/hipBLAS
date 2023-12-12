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
 *
 * ************************************************************************/

#include "hipblas.h"
#include "hipblas.hpp"

#ifndef WIN32
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif

#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to ROCBLAS C89 interfaces
*/

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

#ifdef HIPBLAS_V2
// axpy
hipblasStatus_t hipblasCaxpyCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasCaxpy(
        handle, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZaxpyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZaxpy(handle,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (hipDoubleComplex*)y,
                        incy);
}

// axpy_batched
hipblasStatus_t hipblasCaxpyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasCaxpyBatched(handle,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZaxpyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZaxpyBatched(handle,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// axpy_strided_batched
hipblasStatus_t hipblasCaxpyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasCaxpyStridedBatched(handle,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

hipblasStatus_t hipblasZaxpyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count)
{
    return hipblasZaxpyStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// swap
hipblasStatus_t hipblasCswapCast(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return hipblasCswap(handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZswapCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy)
{
    return hipblasZswap(handle, n, (hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy);
}

// swap_batched
hipblasStatus_t hipblasCswapBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        int                   batch_count)
{
    return hipblasCswapBatched(
        handle, n, (hipComplex* const*)x, incx, (hipComplex* const*)y, incy, batch_count);
}

hipblasStatus_t hipblasZswapBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasZswapBatched(handle,
                               n,
                               (hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// swap_strided_batched
hipblasStatus_t hipblasCswapStridedBatchedCast(hipblasHandle_t handle,
                                               int             n,
                                               hipblasComplex* x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               hipblasComplex* y,
                                               int             incy,
                                               hipblasStride   stridey,
                                               int             batch_count)
{
    return hipblasCswapStridedBatched(
        handle, n, (hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, batch_count);
}

hipblasStatus_t hipblasZswapStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               hipblasDoubleComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasDoubleComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasZswapStridedBatched(handle,
                                      n,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// copy
hipblasStatus_t hipblasCcopyCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return hipblasCcopy(handle, n, (const hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZcopyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZcopy(handle, n, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy);
}

// batched
hipblasStatus_t hipblasCcopyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasCcopyBatched(
        handle, n, (const hipComplex* const*)x, incx, (hipComplex* const*)y, incy, batch_count);
}

hipblasStatus_t hipblasZcopyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZcopyBatched(handle,
                               n,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// strided_batched
hipblasStatus_t hipblasCcopyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasCcopyStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, batch_count);
}

hipblasStatus_t hipblasZcopyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count)
{
    return hipblasZcopyStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// dot
hipblasStatus_t hipblasCdotuCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result)
{
    return hipblasCdotu(
        handle, n, (const hipComplex*)x, incx, (const hipComplex*)y, incy, (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result)
{
    return hipblasZdotu(handle,
                        n,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasCdotcCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result)
{
    return hipblasCdotc(
        handle, n, (const hipComplex*)x, incx, (const hipComplex*)y, incy, (hipComplex*)result);
}

hipblasStatus_t hipblasZdotcCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result)
{
    return hipblasZdotc(handle,
                        n,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)result);
}

// dot_batched
hipblasStatus_t hipblasCdotuBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result)
{
    return hipblasCdotuBatched(handle,
                               n,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               batch_count,
                               (hipComplex*)result);
}

hipblasStatus_t hipblasCdotcBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result)
{
    return hipblasCdotcBatched(handle,
                               n,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               batch_count,
                               (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result)
{
    return hipblasZdotuBatched(handle,
                               n,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               batch_count,
                               (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasZdotcBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result)
{
    return hipblasZdotcBatched(handle,
                               n,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               batch_count,
                               (hipDoubleComplex*)result);
}

// dot_strided_batched
hipblasStatus_t hipblasCdotuStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result)
{
    return hipblasCdotuStridedBatched(handle,
                                      n,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipComplex*)result);
}

hipblasStatus_t hipblasCdotcStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result)
{
    return hipblasCdotcStridedBatched(handle,
                                      n,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result)
{
    return hipblasZdotuStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasZdotcStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result)
{
    return hipblasZdotcStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipDoubleComplex*)result);
}

// asum
hipblasStatus_t hipblasScasumCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{
    return hipblasScasum(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasDzasumCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{
    return hipblasDzasum(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// asum_batched
hipblasStatus_t hipblasScasumBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result)
{
    return hipblasScasumBatched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasDzasumBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result)
{
    return hipblasDzasumBatched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// asum_strided_batched
hipblasStatus_t hipblasScasumStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result)
{
    return hipblasScasumStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasDzasumStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result)
{
    return hipblasDzasumStridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// nrm2
hipblasStatus_t hipblasScnrm2Cast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{
    return hipblasScnrm2(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasDznrm2Cast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{
    return hipblasDznrm2(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// nrm2_batched
hipblasStatus_t hipblasScnrm2BatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result)
{
    return hipblasScnrm2Batched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasDznrm2BatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result)
{
    return hipblasDznrm2Batched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// nrm2_strided_batched
hipblasStatus_t hipblasScnrm2StridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result)
{
    return hipblasScnrm2StridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasDznrm2StridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result)
{
    return hipblasDznrm2StridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// rot
hipblasStatus_t hipblasCrotCast(hipblasHandle_t       handle,
                                int                   n,
                                hipblasComplex*       x,
                                int                   incx,
                                hipblasComplex*       y,
                                int                   incy,
                                const float*          c,
                                const hipblasComplex* s)
{
    return hipblasCrot(
        handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy, c, (const hipComplex*)s);
}

hipblasStatus_t hipblasCsrotCast(hipblasHandle_t handle,
                                 int             n,
                                 hipblasComplex* x,
                                 int             incx,
                                 hipblasComplex* y,
                                 int             incy,
                                 const float*    c,
                                 const float*    s)
{
    return hipblasCsrot(handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy, c, s);
}

hipblasStatus_t hipblasZrotCast(hipblasHandle_t             handle,
                                int                         n,
                                hipblasDoubleComplex*       x,
                                int                         incx,
                                hipblasDoubleComplex*       y,
                                int                         incy,
                                const double*               c,
                                const hipblasDoubleComplex* s)
{
    return hipblasZrot(handle,
                       n,
                       (hipDoubleComplex*)x,
                       incx,
                       (hipDoubleComplex*)y,
                       incy,
                       c,
                       (const hipDoubleComplex*)s);
}

hipblasStatus_t hipblasZdrotCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy,
                                 const double*         c,
                                 const double*         s)
{
    return hipblasZdrot(handle, n, (hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy, c, s);
}

// rot_batched
hipblasStatus_t hipblasCrotBatchedCast(hipblasHandle_t       handle,
                                       int                   n,
                                       hipblasComplex* const x[],
                                       int                   incx,
                                       hipblasComplex* const y[],
                                       int                   incy,
                                       const float*          c,
                                       const hipblasComplex* s,
                                       int                   batch_count)
{
    return hipblasCrotBatched(handle,
                              n,
                              (hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)y,
                              incy,
                              c,
                              (const hipComplex*)s,
                              batch_count);
}

hipblasStatus_t hipblasCsrotBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        const float*          c,
                                        const float*          s,
                                        int                   batch_count)
{
    return hipblasCsrotBatched(
        handle, n, (hipComplex* const*)x, incx, (hipComplex* const*)y, incy, c, s, batch_count);
}

hipblasStatus_t hipblasZrotBatchedCast(hipblasHandle_t             handle,
                                       int                         n,
                                       hipblasDoubleComplex* const x[],
                                       int                         incx,
                                       hipblasDoubleComplex* const y[],
                                       int                         incy,
                                       const double*               c,
                                       const hipblasDoubleComplex* s,
                                       int                         batch_count)
{
    return hipblasZrotBatched(handle,
                              n,
                              (hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)y,
                              incy,
                              c,
                              (const hipDoubleComplex*)s,
                              batch_count);
}

hipblasStatus_t hipblasZdrotBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        const double*               c,
                                        const double*               s,
                                        int                         batch_count)
{
    return hipblasZdrotBatched(handle,
                               n,
                               (hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               c,
                               s,
                               batch_count);
}

// rot_strided_batched
hipblasStatus_t hipblasCrotStridedBatchedCast(hipblasHandle_t       handle,
                                              int                   n,
                                              hipblasComplex*       x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       y,
                                              int                   incy,
                                              hipblasStride         stridey,
                                              const float*          c,
                                              const hipblasComplex* s,
                                              int                   batch_count)
{
    return hipblasCrotStridedBatched(handle,
                                     n,
                                     (hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)y,
                                     incy,
                                     stridey,
                                     c,
                                     (const hipComplex*)s,
                                     batch_count);
}

hipblasStatus_t hipblasCsrotStridedBatchedCast(hipblasHandle_t handle,
                                               int             n,
                                               hipblasComplex* x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               hipblasComplex* y,
                                               int             incy,
                                               hipblasStride   stridey,
                                               const float*    c,
                                               const float*    s,
                                               int             batch_count)
{
    return hipblasCsrotStridedBatched(
        handle, n, (hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, c, s, batch_count);
}

hipblasStatus_t hipblasZrotStridedBatchedCast(hipblasHandle_t             handle,
                                              int                         n,
                                              hipblasDoubleComplex*       x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       y,
                                              int                         incy,
                                              hipblasStride               stridey,
                                              const double*               c,
                                              const hipblasDoubleComplex* s,
                                              int                         batch_count)
{
    return hipblasZrotStridedBatched(handle,
                                     n,
                                     (hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)y,
                                     incy,
                                     stridey,
                                     c,
                                     (const hipDoubleComplex*)s,
                                     batch_count);
}

hipblasStatus_t hipblasZdrotStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               hipblasDoubleComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasDoubleComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               const double*         c,
                                               const double*         s,
                                               int                   batch_count)
{
    return hipblasZdrotStridedBatched(handle,
                                      n,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      c,
                                      s,
                                      batch_count);
}

// rotg
hipblasStatus_t hipblasCrotgCast(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
{
    return hipblasCrotg(handle, (hipComplex*)a, (hipComplex*)b, c, (hipComplex*)s);
}

hipblasStatus_t hipblasZrotgCast(hipblasHandle_t       handle,
                                 hipblasDoubleComplex* a,
                                 hipblasDoubleComplex* b,
                                 double*               c,
                                 hipblasDoubleComplex* s)
{
    return hipblasZrotg(
        handle, (hipDoubleComplex*)a, (hipDoubleComplex*)b, c, (hipDoubleComplex*)s);
}

// rotg_batched
hipblasStatus_t hipblasCrotgBatchedCast(hipblasHandle_t       handle,
                                        hipblasComplex* const a[],
                                        hipblasComplex* const b[],
                                        float* const          c[],
                                        hipblasComplex* const s[],
                                        int                   batch_count)
{
    return hipblasCrotgBatched(handle,
                               (hipComplex* const*)a,
                               (hipComplex* const*)b,
                               c,
                               (hipComplex* const*)s,
                               batch_count);
}

hipblasStatus_t hipblasZrotgBatchedCast(hipblasHandle_t             handle,
                                        hipblasDoubleComplex* const a[],
                                        hipblasDoubleComplex* const b[],
                                        double* const               c[],
                                        hipblasDoubleComplex* const s[],
                                        int                         batch_count)
{
    return hipblasZrotgBatched(handle,
                               (hipDoubleComplex* const*)a,
                               (hipDoubleComplex* const*)b,
                               c,
                               (hipDoubleComplex* const*)s,
                               batch_count);
}

// rotg_strided_batched
hipblasStatus_t hipblasCrotgStridedBatchedCast(hipblasHandle_t handle,
                                               hipblasComplex* a,
                                               hipblasStride   stridea,
                                               hipblasComplex* b,
                                               hipblasStride   strideb,
                                               float*          c,
                                               hipblasStride   stridec,
                                               hipblasComplex* s,
                                               hipblasStride   strides,
                                               int             batch_count)
{
    return hipblasCrotgStridedBatched(handle,
                                      (hipComplex*)a,
                                      stridea,
                                      (hipComplex*)b,
                                      strideb,
                                      c,
                                      stridec,
                                      (hipComplex*)s,
                                      strides,
                                      batch_count);
}

hipblasStatus_t hipblasZrotgStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasDoubleComplex* a,
                                               hipblasStride         stridea,
                                               hipblasDoubleComplex* b,
                                               hipblasStride         strideb,
                                               double*               c,
                                               hipblasStride         stridec,
                                               hipblasDoubleComplex* s,
                                               hipblasStride         strides,
                                               int                   batch_count)
{
    return hipblasZrotgStridedBatched(handle,
                                      (hipDoubleComplex*)a,
                                      stridea,
                                      (hipDoubleComplex*)b,
                                      strideb,
                                      c,
                                      stridec,
                                      (hipDoubleComplex*)s,
                                      strides,
                                      batch_count);
}

// rotm, rotmg - no complex versions

// amax
hipblasStatus_t
    hipblasIcamaxCast(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return hipblasIcamax(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasIzamaxCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamax(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// amax_batched
hipblasStatus_t hipblasIcamaxBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result)
{
    return hipblasIcamaxBatched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIzamaxBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result)
{
    return hipblasIzamaxBatched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// amax_strided_batched
hipblasStatus_t hipblasIcamaxStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result)
{
    return hipblasIcamaxStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIzamaxStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result)
{
    return hipblasIzamaxStridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// amin
hipblasStatus_t
    hipblasIcaminCast(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return hipblasIcamin(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasIzaminCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamin(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// amin_batched
hipblasStatus_t hipblasIcaminBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result)
{
    return hipblasIcaminBatched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIzaminBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result)
{
    return hipblasIzaminBatched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// amin_strided_batched
hipblasStatus_t hipblasIcaminStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result)
{
    return hipblasIcaminStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIzaminStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result)
{
    return hipblasIzaminStridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasCscalCast(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx)
{
    return hipblasCscal(handle, n, (const hipComplex*)alpha, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasCsscalCast(
    hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx)
{
    return hipblasCsscal(handle, n, alpha, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZscalCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZscal(handle, n, (const hipDoubleComplex*)alpha, (hipDoubleComplex*)x, incx);
}

hipblasStatus_t hipblasZdscalCast(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx)
{
    return hipblasZdscal(handle, n, alpha, (hipDoubleComplex*)x, incx);
}

// batched
hipblasStatus_t hipblasCscalBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        const hipblasComplex* alpha,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        int                   batch_count)
{
    return hipblasCscalBatched(
        handle, n, (const hipComplex*)alpha, (hipComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasCsscalBatchedCast(hipblasHandle_t       handle,
                                         int                   n,
                                         const float*          alpha,
                                         hipblasComplex* const x[],
                                         int                   incx,
                                         int                   batch_count)
{
    return hipblasCsscalBatched(handle, n, alpha, (hipComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasZscalBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasDoubleComplex* alpha,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        int                         batch_count)
{
    return hipblasZscalBatched(
        handle, n, (const hipDoubleComplex*)alpha, (hipDoubleComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasZdscalBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const double*               alpha,
                                         hipblasDoubleComplex* const x[],
                                         int                         incx,
                                         int                         batch_count)
{
    return hipblasZdscalBatched(handle, n, alpha, (hipDoubleComplex* const*)x, incx, batch_count);
}

// strided_batched
hipblasStatus_t hipblasCscalStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batch_count)
{
    return hipblasCscalStridedBatched(
        handle, n, (const hipComplex*)alpha, (hipComplex*)x, incx, stridex, batch_count);
}

hipblasStatus_t hipblasCsscalStridedBatchedCast(hipblasHandle_t handle,
                                                int             n,
                                                const float*    alpha,
                                                hipblasComplex* x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                int             batch_count)
{
    return hipblasCsscalStridedBatched(
        handle, n, alpha, (hipComplex*)x, incx, stridex, batch_count);
}

hipblasStatus_t hipblasZscalStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batch_count)
{
    return hipblasZscalStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batch_count);
}

hipblasStatus_t hipblasZdscalStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const double*         alpha,
                                                hipblasDoubleComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count)
{
    return hipblasZdscalStridedBatched(
        handle, n, alpha, (hipDoubleComplex*)x, incx, stridex, batch_count);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
hipblasStatus_t hipblasCgbmvCast(hipblasHandle_t       handle,
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
    return hipblasCgbmv(handle,
                        transA,
                        m,
                        n,
                        kl,
                        ku,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZgbmvCast(hipblasHandle_t             handle,
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
    return hipblasZgbmv(handle,
                        transA,
                        m,
                        n,
                        kl,
                        ku,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// gbmv_batched
hipblasStatus_t hipblasCgbmvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCgbmvBatched(handle,
                               transA,
                               m,
                               n,
                               kl,
                               ku,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZgbmvBatchedCast(hipblasHandle_t                   handle,
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
                                        int                               batch_count)
{
    return hipblasZgbmvBatched(handle,
                               transA,
                               m,
                               n,
                               kl,
                               ku,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// gbmv_strided_batched
hipblasStatus_t hipblasCgbmvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

hipblasStatus_t hipblasZgbmvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

// gemv
hipblasStatus_t hipblasCgemvCast(hipblasHandle_t       handle,
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
    return hipblasCgemv(handle,
                        transA,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZgemvCast(hipblasHandle_t             handle,
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
    return hipblasZgemv(handle,
                        transA,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// gemv_batched
hipblasStatus_t hipblasCgemvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCgemvBatched(handle,
                               transA,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZgemvBatchedCast(hipblasHandle_t                   handle,
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
                                        int                               batch_count)
{
    return hipblasZgemvBatched(handle,
                               transA,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// gemv_strided_batched
hipblasStatus_t hipblasCgemvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

hipblasStatus_t hipblasZgemvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// ger
hipblasStatus_t hipblasCgeruCast(hipblasHandle_t       handle,
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
    return hipblasCgeru(handle,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasCgercCast(hipblasHandle_t       handle,
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
    return hipblasCgerc(handle,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZgeruCast(hipblasHandle_t             handle,
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
    return hipblasZgeru(handle,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZgercCast(hipblasHandle_t             handle,
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
    return hipblasZgerc(handle,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

// ger_batched
hipblasStatus_t hipblasCgeruBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCgeruBatched(handle,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batch_count);
}

hipblasStatus_t hipblasCgercBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCgercBatched(handle,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batch_count);
}

hipblasStatus_t hipblasZgeruBatchedCast(hipblasHandle_t                   handle,
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
    return hipblasZgeruBatched(handle,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batch_count);
}

hipblasStatus_t hipblasZgercBatchedCast(hipblasHandle_t                   handle,
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
    return hipblasZgercBatched(handle,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batch_count);
}

// ger_strided_batched
hipblasStatus_t hipblasCgeruStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCgeruStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

hipblasStatus_t hipblasCgercStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCgercStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

hipblasStatus_t hipblasZgeruStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZgeruStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

hipblasStatus_t hipblasZgercStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZgercStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

// hbmv
hipblasStatus_t hipblasChbmvCast(hipblasHandle_t       handle,
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
    return hipblasChbmv(handle,
                        uplo,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZhbmvCast(hipblasHandle_t             handle,
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
    return hipblasZhbmv(handle,
                        uplo,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// hbmv_batched
hipblasStatus_t hipblasChbmvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasChbmvBatched(handle,
                               uplo,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batchCount);
}

hipblasStatus_t hipblasZhbmvBatchedCast(hipblasHandle_t                   handle,
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
                                        int                               batchCount)
{
    return hipblasZhbmvBatched(handle,
                               uplo,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batchCount);
}

// hbmv_strided_batched
hipblasStatus_t hipblasChbmvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batchCount)
{
    return hipblasChbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

hipblasStatus_t hipblasZhbmvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batchCount)
{
    return hipblasZhbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

// hemv
hipblasStatus_t hipblasChemvCast(hipblasHandle_t       handle,
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
    return hipblasChemv(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZhemvCast(hipblasHandle_t             handle,
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
    return hipblasZhemv(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// hemv_batched
hipblasStatus_t hipblasChemvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasChemvBatched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZhemvBatchedCast(hipblasHandle_t                   handle,
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
                                        int                               batch_count)
{
    return hipblasZhemvBatched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// hemv_strided_batched
hipblasStatus_t hipblasChemvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasChemvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

hipblasStatus_t hipblasZhemvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZhemvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

// her
hipblasStatus_t hipblasCherCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda)
{
    return hipblasCher(handle, uplo, n, alpha, (const hipComplex*)x, incx, (hipComplex*)A, lda);
}

hipblasStatus_t hipblasZherCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda)
{
    return hipblasZher(
        handle, uplo, n, alpha, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)A, lda);
}

// her_batched
hipblasStatus_t hipblasCherBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batchCount)
{
    return hipblasCherBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)A,
                              lda,
                              batchCount);
}

hipblasStatus_t hipblasZherBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batchCount)
{
    return hipblasZherBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)A,
                              lda,
                              batchCount);
}

// her_strided_batched
hipblasStatus_t hipblasCherStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const float*          alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       A,
                                              int                   lda,
                                              hipblasStride         strideA,
                                              int                   batchCount)
{
    return hipblasCherStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)A,
                                     lda,
                                     strideA,
                                     batchCount);
}

hipblasStatus_t hipblasZherStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const double*               alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       A,
                                              int                         lda,
                                              hipblasStride               strideA,
                                              int                         batchCount)
{
    return hipblasZherStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)A,
                                     lda,
                                     strideA,
                                     batchCount);
}

// her2
hipblasStatus_t hipblasCher2Cast(hipblasHandle_t       handle,
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
    return hipblasCher2(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZher2Cast(hipblasHandle_t             handle,
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
    return hipblasZher2(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

// her2_batched
hipblasStatus_t hipblasCher2BatchedCast(hipblasHandle_t             handle,
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
    return hipblasCher2Batched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batchCount);
}

hipblasStatus_t hipblasZher2BatchedCast(hipblasHandle_t                   handle,
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
    return hipblasZher2Batched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batchCount);
}

// her2_strided_batched
hipblasStatus_t hipblasCher2StridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batchCount)
{
    return hipblasCher2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

hipblasStatus_t hipblasZher2StridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batchCount)
{
    return hipblasZher2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

// hpmv
hipblasStatus_t hipblasChpmvCast(hipblasHandle_t       handle,
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
    return hipblasChpmv(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)AP,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZhpmvCast(hipblasHandle_t             handle,
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
    return hipblasZhpmv(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)AP,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// hpmv_batched
hipblasStatus_t hipblasChpmvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasChpmvBatched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)AP,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batchCount);
}

hipblasStatus_t hipblasZhpmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const AP[],
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batchCount)
{
    return hipblasZhpmvBatched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)AP,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batchCount);
}

// hpmv_strided_batched
hipblasStatus_t hipblasChpmvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batchCount)
{
    return hipblasChpmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)AP,
                                      strideAP,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

hipblasStatus_t hipblasZhpmvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batchCount)
{
    return hipblasZhpmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)AP,
                                      strideAP,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

// hpr
hipblasStatus_t hipblasChprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP)
{
    return hipblasChpr(handle, uplo, n, alpha, (const hipComplex*)x, incx, (hipComplex*)AP);
}

hipblasStatus_t hipblasZhprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP)
{
    return hipblasZhpr(
        handle, uplo, n, alpha, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)AP);
}

// hpr_batched
hipblasStatus_t hipblasChprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount)
{
    return hipblasChprBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)AP,
                              batchCount);
}

hipblasStatus_t hipblasZhprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount)
{
    return hipblasZhprBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)AP,
                              batchCount);
}

// hpr_strided_batched
hipblasStatus_t hipblasChprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const float*          alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount)
{
    return hipblasChprStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)AP,
                                     strideAP,
                                     batchCount);
}

hipblasStatus_t hipblasZhprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const double*               alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount)
{
    return hipblasZhprStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)AP,
                                     strideAP,
                                     batchCount);
}

// hpr2
hipblasStatus_t hipblasChpr2Cast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       AP)
{
    return hipblasChpr2(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)AP);
}

hipblasStatus_t hipblasZhpr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       AP)
{
    return hipblasZhpr2(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)AP);
}

// hpr2_batched
hipblasStatus_t hipblasChpr2BatchedCast(hipblasHandle_t             handle,
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
    return hipblasChpr2Batched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)AP,
                               batchCount);
}

hipblasStatus_t hipblasZhpr2BatchedCast(hipblasHandle_t                   handle,
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
    return hipblasZhpr2Batched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)AP,
                               batchCount);
}

// hpr2_strided_batched
hipblasStatus_t hipblasChpr2StridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batchCount)
{
    return hipblasChpr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)AP,
                                      strideAP,
                                      batchCount);
}

hipblasStatus_t hipblasZhpr2StridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batchCount)
{
    return hipblasZhpr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)AP,
                                      strideAP,
                                      batchCount);
}

// sbmv, spmv, spr2 no complex versions

// spr
hipblasStatus_t hipblasCsprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP)
{
    return hipblasCspr(
        handle, uplo, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)AP);
}

hipblasStatus_t hipblasZsprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP)
{
    return hipblasZspr(handle,
                       uplo,
                       n,
                       (const hipDoubleComplex*)alpha,
                       (const hipDoubleComplex*)x,
                       incx,
                       (hipDoubleComplex*)AP);
}

// spr_batched
hipblasStatus_t hipblasCsprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount)
{
    return hipblasCsprBatched(handle,
                              uplo,
                              n,
                              (const hipComplex*)alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)AP,
                              batchCount);
}

hipblasStatus_t hipblasZsprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount)
{
    return hipblasZsprBatched(handle,
                              uplo,
                              n,
                              (const hipDoubleComplex*)alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)AP,
                              batchCount);
}

// spr_strided_batched
hipblasStatus_t hipblasCsprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const hipblasComplex* alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount)
{
    return hipblasCsprStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipComplex*)alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)AP,
                                     strideAP,
                                     batchCount);
}

hipblasStatus_t hipblasZsprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const hipblasDoubleComplex* alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount)
{
    return hipblasZsprStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipDoubleComplex*)alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)AP,
                                     strideAP,
                                     batchCount);
}

// symv
hipblasStatus_t hipblasCsymvCast(hipblasHandle_t       handle,
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
    return hipblasCsymv(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZsymvCast(hipblasHandle_t             handle,
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
    return hipblasZsymv(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// symv_batched
hipblasStatus_t hipblasCsymvBatchedCast(hipblasHandle_t             handle,
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
                                        int                         batchCount)
{
    return hipblasCsymvBatched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batchCount);
}

hipblasStatus_t hipblasZsymvBatchedCast(hipblasHandle_t                   handle,
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
                                        int                               batchCount)
{
    return hipblasZsymvBatched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batchCount);
}

// symv_strided_batched
hipblasStatus_t hipblasCsymvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batchCount)
{
    return hipblasCsymvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

hipblasStatus_t hipblasZsymvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batchCount)
{
    return hipblasZsymvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

// syr
hipblasStatus_t hipblasCsyrCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda)
{
    return hipblasCsyr(
        handle, uplo, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)A, lda);
}

hipblasStatus_t hipblasZsyrCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda)
{
    return hipblasZsyr(handle,
                       uplo,
                       n,
                       (const hipDoubleComplex*)alpha,
                       (const hipDoubleComplex*)x,
                       incx,
                       (hipDoubleComplex*)A,
                       lda);
}

// syr_batched
hipblasStatus_t hipblasCsyrBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batch_count)
{
    return hipblasCsyrBatched(handle,
                              uplo,
                              n,
                              (const hipComplex*)alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)A,
                              lda,
                              batch_count);
}

hipblasStatus_t hipblasZsyrBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batch_count)
{
    return hipblasZsyrBatched(handle,
                              uplo,
                              n,
                              (const hipDoubleComplex*)alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)A,
                              lda,
                              batch_count);
}

// syr_strided_batched
hipblasStatus_t hipblasCsyrStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const hipblasComplex* alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       A,
                                              int                   lda,
                                              hipblasStride         strideA,
                                              int                   batch_count)
{
    return hipblasCsyrStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipComplex*)alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)A,
                                     lda,
                                     strideA,
                                     batch_count);
}

hipblasStatus_t hipblasZsyrStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const hipblasDoubleComplex* alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       A,
                                              int                         lda,
                                              hipblasStride               strideA,
                                              int                         batch_count)
{
    return hipblasZsyrStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipDoubleComplex*)alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)A,
                                     lda,
                                     strideA,
                                     batch_count);
}

// syr2
hipblasStatus_t hipblasCsyr2Cast(hipblasHandle_t       handle,
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
    return hipblasCsyr2(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZsyr2Cast(hipblasHandle_t             handle,
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
    return hipblasZsyr2(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

// syr2_batched
hipblasStatus_t hipblasCsyr2BatchedCast(hipblasHandle_t             handle,
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
    return hipblasCsyr2Batched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batchCount);
}

hipblasStatus_t hipblasZsyr2BatchedCast(hipblasHandle_t                   handle,
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
    return hipblasZsyr2Batched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batchCount);
}

// syr2_strided_batched
hipblasStatus_t hipblasCsyr2StridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batchCount)
{
    return hipblasCsyr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

hipblasStatus_t hipblasZsyr2StridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batchCount)
{
    return hipblasZsyr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

// trsv
hipblasStatus_t hipblasCtrsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtrsv(
        handle, uplo, transA, diag, m, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtrsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtrsv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)x, incx);
}

// trsv_batched
hipblasStatus_t hipblasCtrsvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCtrsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batch_count);
}

hipblasStatus_t hipblasZtrsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count)
{
    return hipblasZtrsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batch_count);
}

// trsv_strided_batched
hipblasStatus_t hipblasCtrsvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCtrsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batch_count);
}

hipblasStatus_t hipblasZtrsvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZtrsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batch_count);
}

// tbmv
hipblasStatus_t hipblasCtbmvCast(hipblasHandle_t       handle,
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
    return hipblasCtbmv(
        handle, uplo, transA, diag, m, k, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtbmvCast(hipblasHandle_t             handle,
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
    return hipblasZtbmv(handle,
                        uplo,
                        transA,
                        diag,
                        m,
                        k,
                        (const hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)x,
                        incx);
}

// tbmv_batched
hipblasStatus_t hipblasCtbmvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCtbmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               k,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batch_count);
}

hipblasStatus_t hipblasZtbmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        int                               k,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count)
{
    return hipblasZtbmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               k,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batch_count);
}

// tbmv_strided_batched
hipblasStatus_t hipblasCtbmvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCtbmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      k,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

hipblasStatus_t hipblasZtbmvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZtbmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      k,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

// tbsv
hipblasStatus_t hipblasCtbsvCast(hipblasHandle_t       handle,
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
    return hipblasCtbsv(
        handle, uplo, transA, diag, n, k, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtbsvCast(hipblasHandle_t             handle,
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
    return hipblasZtbsv(handle,
                        uplo,
                        transA,
                        diag,
                        n,
                        k,
                        (const hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)x,
                        incx);
}

// tbsv_batched
hipblasStatus_t hipblasCtbsvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCtbsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               n,
                               k,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batchCount);
}

hipblasStatus_t hipblasZtbsvBatchedCast(hipblasHandle_t                   handle,
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
    return hipblasZtbsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               n,
                               k,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batchCount);
}

// tbsv_strided_batched
hipblasStatus_t hipblasCtbsvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   n,
                                               int                   k,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batchCount)
{
    return hipblasCtbsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      n,
                                      k,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

hipblasStatus_t hipblasZtbsvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         n,
                                               int                         k,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batchCount)
{
    return hipblasZtbsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

// tpmv
hipblasStatus_t hipblasCtpmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtpmv(handle, uplo, transA, diag, m, (const hipComplex*)AP, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtpmv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)AP, (hipDoubleComplex*)x, incx);
}

// tpmv_batched
hipblasStatus_t hipblasCtpmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount)
{
    return hipblasCtpmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)AP,
                               (hipComplex* const*)x,
                               incx,
                               batchCount);
}

hipblasStatus_t hipblasZtpmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount)
{
    return hipblasZtpmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)AP,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batchCount);
}

// tpmv_strided_batched
hipblasStatus_t hipblasCtpmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               const hipblasComplex* AP,
                                               hipblasStride         strideAP,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batchCount)
{
    return hipblasCtpmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)AP,
                                      strideAP,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

hipblasStatus_t hipblasZtpmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               const hipblasDoubleComplex* AP,
                                               hipblasStride               strideAP,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batchCount)
{
    return hipblasZtpmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)AP,
                                      strideAP,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

// tpsv
hipblasStatus_t hipblasCtpsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtpsv(handle, uplo, transA, diag, m, (const hipComplex*)AP, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtpsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtpsv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)AP, (hipDoubleComplex*)x, incx);
}

// tpsv_batched
hipblasStatus_t hipblasCtpsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount)
{
    return hipblasCtpsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)AP,
                               (hipComplex* const*)x,
                               incx,
                               batchCount);
}

hipblasStatus_t hipblasZtpsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount)
{
    return hipblasZtpsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)AP,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batchCount);
}

// tpsv_strided_batched
hipblasStatus_t hipblasCtpsvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               const hipblasComplex* AP,
                                               hipblasStride         strideAP,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batchCount)
{
    return hipblasCtpsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)AP,
                                      strideAP,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

hipblasStatus_t hipblasZtpsvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               const hipblasDoubleComplex* AP,
                                               hipblasStride               strideAP,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batchCount)
{
    return hipblasZtpsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)AP,
                                      strideAP,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

// trmv
hipblasStatus_t hipblasCtrmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtrmv(
        handle, uplo, transA, diag, m, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtrmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtrmv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)x, incx);
}

// trmv_batched
hipblasStatus_t hipblasCtrmvBatchedCast(hipblasHandle_t             handle,
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
    return hipblasCtrmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batch_count);
}

hipblasStatus_t hipblasZtrmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count)
{
    return hipblasZtrmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batch_count);
}

// trmv_strided_batched
hipblasStatus_t hipblasCtrmvStridedBatchedCast(hipblasHandle_t       handle,
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
                                               int                   batch_count)
{
    return hipblasCtrmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

hipblasStatus_t hipblasZtrmvStridedBatchedCast(hipblasHandle_t             handle,
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
                                               int                         batch_count)
{
    return hipblasZtrmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

#endif

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// trtri
template <>
hipblasStatus_t hipblasTrtri<float>(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    hipblasDiagType_t diag,
                                    int               n,
                                    const float*      A,
                                    int               lda,
                                    float*            invA,
                                    int               ldinvA)
{
    return hipblasStrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
hipblasStatus_t hipblasTrtri<double>(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     hipblasDiagType_t diag,
                                     int               n,
                                     const double*     A,
                                     int               lda,
                                     double*           invA,
                                     int               ldinvA)
{
    return hipblasDtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
hipblasStatus_t hipblasTrtri<hipblasComplex>(hipblasHandle_t       handle,
                                             hipblasFillMode_t     uplo,
                                             hipblasDiagType_t     diag,
                                             int                   n,
                                             const hipblasComplex* A,
                                             int                   lda,
                                             hipblasComplex*       invA,
                                             int                   ldinvA)
{
#ifdef HIPBLAS_V2
    return hipblasCtrtri(
        handle, uplo, diag, n, (const hipComplex*)A, lda, (hipComplex*)invA, ldinvA);
#else
    return hipblasCtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
#endif
}

template <>
hipblasStatus_t hipblasTrtri<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasDiagType_t           diag,
                                                   int                         n,
                                                   const hipblasDoubleComplex* A,
                                                   int                         lda,
                                                   hipblasDoubleComplex*       invA,
                                                   int                         ldinvA)
{
#ifdef HIPBLAS_V2
    return hipblasZtrtri(
        handle, uplo, diag, n, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)invA, ldinvA);
#else
    return hipblasZtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
#endif
}

// trtri_batched
template <>
hipblasStatus_t hipblasTrtriBatched<float>(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const float* const A[],
                                           int                lda,
                                           float*             invA[],
                                           int                ldinvA,
                                           int                batch_count)
{
    return hipblasStrtriBatched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriBatched<double>(hipblasHandle_t     handle,
                                            hipblasFillMode_t   uplo,
                                            hipblasDiagType_t   diag,
                                            int                 n,
                                            const double* const A[],
                                            int                 lda,
                                            double*             invA[],
                                            int                 ldinvA,
                                            int                 batch_count)
{
    return hipblasDtrtriBatched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriBatched<hipblasComplex>(hipblasHandle_t             handle,
                                                    hipblasFillMode_t           uplo,
                                                    hipblasDiagType_t           diag,
                                                    int                         n,
                                                    const hipblasComplex* const A[],
                                                    int                         lda,
                                                    hipblasComplex*             invA[],
                                                    int                         ldinvA,
                                                    int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrtriBatched(handle,
                                uplo,
                                diag,
                                n,
                                (const hipComplex* const*)A,
                                lda,
                                (hipComplex**)invA,
                                ldinvA,
                                batch_count);
#else
    return hipblasCtrtriBatched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
#endif
}

template <>
hipblasStatus_t hipblasTrtriBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
                                                          hipblasFillMode_t                 uplo,
                                                          hipblasDiagType_t                 diag,
                                                          int                               n,
                                                          const hipblasDoubleComplex* const A[],
                                                          int                               lda,
                                                          hipblasDoubleComplex*             invA[],
                                                          int                               ldinvA,
                                                          int batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrtriBatched(handle,
                                uplo,
                                diag,
                                n,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (hipDoubleComplex**)invA,
                                ldinvA,
                                batch_count);
#else
    return hipblasZtrtriBatched(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
#endif
}

// trtri_strided_batched
template <>
hipblasStatus_t hipblasTrtriStridedBatched<float>(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  hipblasDiagType_t diag,
                                                  int               n,
                                                  const float*      A,
                                                  int               lda,
                                                  hipblasStride     stride_A,
                                                  float*            invA,
                                                  int               ldinvA,
                                                  hipblasStride     stride_invA,
                                                  int               batch_count)
{
    return hipblasStrtriStridedBatched(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriStridedBatched<double>(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   hipblasDiagType_t diag,
                                                   int               n,
                                                   const double*     A,
                                                   int               lda,
                                                   hipblasStride     stride_A,
                                                   double*           invA,
                                                   int               ldinvA,
                                                   hipblasStride     stride_invA,
                                                   int               batch_count)
{
    return hipblasDtrtriStridedBatched(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriStridedBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                           hipblasFillMode_t     uplo,
                                                           hipblasDiagType_t     diag,
                                                           int                   n,
                                                           const hipblasComplex* A,
                                                           int                   lda,
                                                           hipblasStride         stride_A,
                                                           hipblasComplex*       invA,
                                                           int                   ldinvA,
                                                           hipblasStride         stride_invA,
                                                           int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrtriStridedBatched(handle,
                                       uplo,
                                       diag,
                                       n,
                                       (const hipComplex*)A,
                                       lda,
                                       stride_A,
                                       (hipComplex*)invA,
                                       ldinvA,
                                       stride_invA,
                                       batch_count);
#else
    return hipblasCtrtriStridedBatched(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
#endif
}

template <>
hipblasStatus_t hipblasTrtriStridedBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                                 hipblasFillMode_t           uplo,
                                                                 hipblasDiagType_t           diag,
                                                                 int                         n,
                                                                 const hipblasDoubleComplex* A,
                                                                 int                         lda,
                                                                 hipblasStride         stride_A,
                                                                 hipblasDoubleComplex* invA,
                                                                 int                   ldinvA,
                                                                 hipblasStride         stride_invA,
                                                                 int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrtriStridedBatched(handle,
                                       uplo,
                                       diag,
                                       n,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       stride_A,
                                       (hipDoubleComplex*)invA,
                                       ldinvA,
                                       stride_invA,
                                       batch_count);
#else
    return hipblasZtrtriStridedBatched(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
#endif
}

// dgmm
template <>
hipblasStatus_t hipblasDgmm(hipblasHandle_t   handle,
                            hipblasSideMode_t side,
                            int               m,
                            int               n,
                            const float*      A,
                            int               lda,
                            const float*      x,
                            int               incx,
                            float*            C,
                            int               ldc)
{
    return hipblasSdgmm(handle, side, m, n, A, lda, x, incx, C, ldc);
}

template <>
hipblasStatus_t hipblasDgmm(hipblasHandle_t   handle,
                            hipblasSideMode_t side,
                            int               m,
                            int               n,
                            const double*     A,
                            int               lda,
                            const double*     x,
                            int               incx,
                            double*           C,
                            int               ldc)
{
    return hipblasDdgmm(handle, side, m, n, A, lda, x, incx, C, ldc);
}

template <>
hipblasStatus_t hipblasDgmm(hipblasHandle_t       handle,
                            hipblasSideMode_t     side,
                            int                   m,
                            int                   n,
                            const hipblasComplex* A,
                            int                   lda,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       C,
                            int                   ldc)
{
#ifdef HIPBLAS_V2
    return hipblasCdgmm(handle,
                        side,
                        m,
                        n,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasCdgmm(handle, side, m, n, A, lda, x, incx, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasDgmm(hipblasHandle_t             handle,
                            hipblasSideMode_t           side,
                            int                         m,
                            int                         n,
                            const hipblasDoubleComplex* A,
                            int                         lda,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       C,
                            int                         ldc)
{
#ifdef HIPBLAS_V2
    return hipblasZdgmm(handle,
                        side,
                        m,
                        n,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZdgmm(handle, side, m, n, A, lda, x, incx, C, ldc);
#endif
}

// dgmm_batched
template <>
hipblasStatus_t hipblasDgmmBatched(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   int                m,
                                   int                n,
                                   const float* const A[],
                                   int                lda,
                                   const float* const x[],
                                   int                incx,
                                   float* const       C[],
                                   int                ldc,
                                   int                batch_count)
{
    return hipblasSdgmmBatched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmBatched(hipblasHandle_t     handle,
                                   hipblasSideMode_t   side,
                                   int                 m,
                                   int                 n,
                                   const double* const A[],
                                   int                 lda,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       C[],
                                   int                 ldc,
                                   int                 batch_count)
{
    return hipblasDdgmmBatched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmBatched(hipblasHandle_t             handle,
                                   hipblasSideMode_t           side,
                                   int                         m,
                                   int                         n,
                                   const hipblasComplex* const A[],
                                   int                         lda,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       C[],
                                   int                         ldc,
                                   int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCdgmmBatched(handle,
                               side,
                               m,
                               n,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (hipComplex* const*)C,
                               ldc,
                               batch_count);
#else
    return hipblasCdgmmBatched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
#endif
}

template <>
hipblasStatus_t hipblasDgmmBatched(hipblasHandle_t                   handle,
                                   hipblasSideMode_t                 side,
                                   int                               m,
                                   int                               n,
                                   const hipblasDoubleComplex* const A[],
                                   int                               lda,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       C[],
                                   int                               ldc,
                                   int                               batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZdgmmBatched(handle,
                               side,
                               m,
                               n,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batch_count);
#else
    return hipblasZdgmmBatched(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
#endif
}

// dgmm_strided_batched
template <>
hipblasStatus_t hipblasDgmmStridedBatched(hipblasHandle_t   handle,
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
                                          int               batch_count)
{
    return hipblasSdgmmStridedBatched(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmStridedBatched(hipblasHandle_t   handle,
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
                                          int               batch_count)
{
    return hipblasDdgmmStridedBatched(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmStridedBatched(hipblasHandle_t       handle,
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
                                          int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCdgmmStridedBatched(handle,
                                      side,
                                      m,
                                      n,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_A,
                                      (const hipComplex*)x,
                                      incx,
                                      stride_x,
                                      (hipComplex*)C,
                                      ldc,
                                      stride_C,
                                      batch_count);
#else
    return hipblasCdgmmStridedBatched(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
#endif
}

template <>
hipblasStatus_t hipblasDgmmStridedBatched(hipblasHandle_t             handle,
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
                                          int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZdgmmStridedBatched(handle,
                                      side,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_A,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      stride_C,
                                      batch_count);
#else
    return hipblasZdgmmStridedBatched(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
#endif
}

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
#ifdef HIPBLAS_V2
    return hipblasCgemm(handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasCgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgemm(handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgemmBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batch_count);
#else
    return hipblasCgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgemmBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batch_count);
#else
    return hipblasZgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      bsa,
                                      (const hipComplex*)B,
                                      ldb,
                                      bsb,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      bsc,
                                      batch_count);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      bsa,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      bsb,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      bsc,
                                      batch_count);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCherk(
        handle, uplo, transA, n, k, alpha, (const hipComplex*)A, lda, beta, (hipComplex*)C, ldc);
#else
    return hipblasCherk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZherk(handle,
                        uplo,
                        transA,
                        n,
                        k,
                        alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        beta,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZherk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCherkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               alpha,
                               (const hipComplex* const*)A,
                               lda,
                               beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasCherkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZherkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasZherkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
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
                                          hipblasStride         strideA,
                                          const float*          beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          hipblasStride         strideC,
                                          int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCherkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
    return hipblasCherkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
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
                                          hipblasStride               strideA,
                                          const double*               beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          hipblasStride               strideC,
                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZherkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
    return hipblasZherkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCher2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         beta,
                         (hipComplex*)C,
                         ldc);
#else
    return hipblasCher2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZher2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         beta,
                         (hipDoubleComplex*)C,
                         ldc);
#else
    return hipblasZher2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCher2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasCher2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZher2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasZher2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCher2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZher2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCherkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         beta,
                         (hipComplex*)C,
                         ldc);
#else
    return hipblasCherkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZherkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         beta,
                         (hipDoubleComplex*)C,
                         ldc);
#else
    return hipblasZherkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCherkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasCherkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZherkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasZherkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCherkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZherkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsymm(handle,
                        side,
                        uplo,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsymm(handle,
                        side,
                        uplo,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsymmBatched(handle,
                               side,
                               uplo,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasCsymmBatched(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsymmBatched(handle,
                               side,
                               uplo,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasZsymmBatched(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
                                          hipblasStride     strideA,
                                          const float*      B,
                                          int               ldb,
                                          hipblasStride     strideB,
                                          const float*      beta,
                                          float*            C,
                                          int               ldc,
                                          hipblasStride     strideC,
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
                                          hipblasStride     strideA,
                                          const double*     B,
                                          int               ldb,
                                          hipblasStride     strideB,
                                          const double*     beta,
                                          double*           C,
                                          int               ldc,
                                          hipblasStride     strideC,
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
                                          hipblasStride         strideA,
                                          const hipblasComplex* B,
                                          int                   ldb,
                                          hipblasStride         strideB,
                                          const hipblasComplex* beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          hipblasStride         strideC,
                                          int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
#endif
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
                                          hipblasStride               strideA,
                                          const hipblasDoubleComplex* B,
                                          int                         ldb,
                                          hipblasStride               strideB,
                                          const hipblasDoubleComplex* beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          hipblasStride               strideC,
                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsyrk(handle,
                        uplo,
                        transA,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasCsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsyrk(handle,
                        uplo,
                        transA,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZsyrk(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasCsyrkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasZsyrkBatched(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
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
                                          hipblasStride      strideA,
                                          const float*       beta,
                                          float*             C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                          hipblasStride      strideA,
                                          const double*      beta,
                                          double*            C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                          hipblasStride         strideA,
                                          const hipblasComplex* beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          hipblasStride         strideC,
                                          int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsyrkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
    return hipblasCsyrkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
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
                                          hipblasStride               strideA,
                                          const hipblasDoubleComplex* beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          hipblasStride               strideC,
                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsyrkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
    return hipblasZsyrkStridedBatched(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsyr2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         (const hipComplex*)beta,
                         (hipComplex*)C,
                         ldc);
#else
    return hipblasCsyr2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsyr2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         (const hipDoubleComplex*)beta,
                         (hipDoubleComplex*)C,
                         ldc);
#else
    return hipblasZsyr2k(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsyr2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                (const hipComplex*)beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasCsyr2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsyr2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                (const hipDoubleComplex*)beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasZsyr2kBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
                                           hipblasStride      strideA,
                                           const float*       B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                           hipblasStride      strideA,
                                           const double*      B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipComplex*)beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipDoubleComplex*)beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         (const hipComplex*)beta,
                         (hipComplex*)C,
                         ldc);
#else
    return hipblasCsyrkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         (const hipDoubleComplex*)beta,
                         (hipDoubleComplex*)C,
                         ldc);
#else
    return hipblasZsyrkx(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                (const hipComplex*)beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasCsyrkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                (const hipDoubleComplex*)beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
#else
    return hipblasZsyrkxBatched(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
                                           hipblasStride      strideA,
                                           const float*       B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                           hipblasStride      strideA,
                                           const double*      B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                           hipblasStride         strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipComplex*)beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
                                           hipblasStride               strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipDoubleComplex*)beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasChemm(handle,
                        side,
                        uplo,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasChemm(handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZhemm(handle,
                        side,
                        uplo,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZhemm(handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasChemmBatched(handle,
                               side,
                               uplo,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasChemmBatched(
        handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZhemmBatched(handle,
                               side,
                               uplo,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasZhemmBatched(
        handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
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
                                          hipblasStride         strideA,
                                          const hipblasComplex* B,
                                          int                   ldb,
                                          hipblasStride         strideB,
                                          const hipblasComplex* beta,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          hipblasStride         strideC,
                                          int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasChemmStridedBatched(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
#endif
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
                                          hipblasStride               strideA,
                                          const hipblasDoubleComplex* B,
                                          int                         ldb,
                                          hipblasStride               strideB,
                                          const hipblasDoubleComplex* beta,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          hipblasStride               strideC,
                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZhemmStridedBatched(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
#endif
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
                                   const float*       B,
                                   int                ldb,
                                   float*             C,
                                   int                ldc)
{
    return hipblasStrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
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
                                    const double*      B,
                                    int                ldb,
                                    double*            C,
                                    int                ldc)
{
    return hipblasDtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
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
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasComplex*       C,
                                            int                   ldc)
{
#ifdef HIPBLAS_V2
    return hipblasCtrmm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasCtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
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
                                                  const hipblasDoubleComplex* B,
                                                  int                         ldb,
                                                  hipblasDoubleComplex*       C,
                                                  int                         ldc)
{
#ifdef HIPBLAS_V2
    return hipblasZtrmm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZtrmm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
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
                                          const float* const B[],
                                          int                ldb,
                                          float* const       C[],
                                          int                ldc,
                                          int                batchCount)
{
    return hipblasStrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
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
                                           const double* const B[],
                                           int                 ldb,
                                           double* const       C[],
                                           int                 ldc,
                                           int                 batchCount)
{
    return hipblasDtrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
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
                                                   const hipblasComplex* const B[],
                                                   int                         ldb,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCtrmmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasCtrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
#endif
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
                                                         const hipblasDoubleComplex* const B[],
                                                         int                               ldb,
                                                         hipblasDoubleComplex* const       C[],
                                                         int                               ldc,
                                                         int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZtrmmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasZtrmmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
#endif
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
                                                 hipblasStride      strideA,
                                                 const float*       B,
                                                 int                ldb,
                                                 hipblasStride      strideB,
                                                 float*             C,
                                                 int                ldc,
                                                 hipblasStride      strideC,
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
                                      C,
                                      ldc,
                                      strideC,
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
                                                  hipblasStride      strideA,
                                                  const double*      B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  double*            C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
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
                                      C,
                                      ldc,
                                      strideC,
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
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         strideC,
                                                          int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCtrmmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
#endif
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
                                                                hipblasStride               strideA,
                                                                const hipblasDoubleComplex* B,
                                                                int                         ldb,
                                                                hipblasStride               strideB,
                                                                hipblasDoubleComplex*       C,
                                                                int                         ldc,
                                                                hipblasStride               strideC,
                                                                int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZtrmmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount);
#endif
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
                                   const float*       A,
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
                                    const double*      A,
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
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       B,
                                            int                   ldb)
{
#ifdef HIPBLAS_V2
    return hipblasCtrsm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (hipComplex*)B,
                        ldb);
#else
    return hipblasCtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
#endif
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
                                                  const hipblasDoubleComplex* A,
                                                  int                         lda,
                                                  hipblasDoubleComplex*       B,
                                                  int                         ldb)
{
#ifdef HIPBLAS_V2
    return hipblasZtrsm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)B,
                        ldb);
#else
    return hipblasZtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
#endif
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
                                          const float* const A[],
                                          int                lda,
                                          float* const       B[],
                                          int                ldb,
                                          int                batch_count)
{
    return hipblasStrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

template <>
hipblasStatus_t hipblasTrsmBatched<double>(hipblasHandle_t     handle,
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
                                           int                 batch_count)
{
    return hipblasDtrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

template <>
hipblasStatus_t hipblasTrsmBatched<hipblasComplex>(hipblasHandle_t             handle,
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
                                                   int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrsmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)B,
                               ldb,
                               batch_count);
#else
    return hipblasCtrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
#endif
}

template <>
hipblasStatus_t hipblasTrsmBatched<hipblasDoubleComplex>(hipblasHandle_t                   handle,
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
                                                         int batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrsmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)B,
                               ldb,
                               batch_count);
#else
    return hipblasZtrsmBatched(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
#endif
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
                                                 const float*       A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 float*             B,
                                                 int                ldb,
                                                 hipblasStride      strideB,
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
                                                  const double*      A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  double*            B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
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
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          hipblasComplex*       B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrsmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)B,
                                      ldb,
                                      strideB,
                                      batch_count);
#else
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
#endif
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
                                                                const hipblasDoubleComplex* A,
                                                                int                         lda,
                                                                hipblasStride               strideA,
                                                                hipblasDoubleComplex*       B,
                                                                int                         ldb,
                                                                hipblasStride               strideB,
                                                                int batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrsmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      batch_count);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgeam(handle,
                        transA,
                        transB,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)beta,
                        (const hipComplex*)B,
                        ldb,
                        (hipComplex*)C,
                        ldc);
#else
    return hipblasCgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgeam(handle,
                        transA,
                        transB,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)beta,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (hipDoubleComplex*)C,
                        ldc);
#else
    return hipblasZgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgeamBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex*)beta,
                               (const hipComplex* const*)B,
                               ldb,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasCgeamBatched(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgeamBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex*)beta,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
#else
    return hipblasZgeamBatched(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
#endif
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
                                          hipblasStride      strideA,
                                          const float*       beta,
                                          const float*       B,
                                          int                ldb,
                                          hipblasStride      strideB,
                                          float*             C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                          hipblasStride      strideA,
                                          const double*      beta,
                                          const double*      B,
                                          int                ldb,
                                          hipblasStride      strideB,
                                          double*            C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                          hipblasStride         strideA,
                                          const hipblasComplex* beta,
                                          const hipblasComplex* B,
                                          int                   ldb,
                                          hipblasStride         strideB,
                                          hipblasComplex*       C,
                                          int                   ldc,
                                          hipblasStride         strideC,
                                          int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)beta,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
#endif
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
                                          hipblasStride               strideA,
                                          const hipblasDoubleComplex* beta,
                                          const hipblasDoubleComplex* B,
                                          int                         ldb,
                                          hipblasStride               strideB,
                                          hipblasDoubleComplex*       C,
                                          int                         ldc,
                                          hipblasStride               strideC,
                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)beta,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
#else
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
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgetrf(handle, n, (hipComplex*)A, lda, ipiv, info);
#else
    return hipblasCgetrf(handle, n, A, lda, ipiv, info);
#endif
}

template <>
hipblasStatus_t hipblasGetrf<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   int*                  ipiv,
                                                   int*                  info)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrf(handle, n, (hipDoubleComplex*)A, lda, ipiv, info);
#else
    return hipblasZgetrf(handle, n, A, lda, ipiv, info);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgetrfBatched(handle, n, (hipComplex* const*)A, lda, ipiv, info, batchCount);
#else
    return hipblasCgetrfBatched(handle, n, A, lda, ipiv, info, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgetrfBatched(
        handle, n, (hipDoubleComplex* const*)A, lda, ipiv, info, batchCount);
#else
    return hipblasZgetrfBatched(handle, n, A, lda, ipiv, info, batchCount);
#endif
}

// getrf_strided_batched
template <>
hipblasStatus_t hipblasGetrfStridedBatched<float>(hipblasHandle_t     handle,
                                                  const int           n,
                                                  float*              A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  int*                ipiv,
                                                  const hipblasStride strideP,
                                                  int*                info,
                                                  const int           batchCount)
{
    return hipblasSgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<double>(hipblasHandle_t     handle,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride strideA,
                                                   int*                ipiv,
                                                   const hipblasStride strideP,
                                                   int*                info,
                                                   const int           batchCount)
{
    return hipblasDgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<hipblasComplex>(hipblasHandle_t     handle,
                                                           const int           n,
                                                           hipblasComplex*     A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           int*                ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetrfStridedBatched(
        handle, n, (hipComplex*)A, lda, strideA, ipiv, strideP, info, batchCount);
#else
    return hipblasCgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                                 const int             n,
                                                                 hipblasDoubleComplex* A,
                                                                 const int             lda,
                                                                 const hipblasStride   strideA,
                                                                 int*                  ipiv,
                                                                 const hipblasStride   strideP,
                                                                 int*                  info,
                                                                 const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrfStridedBatched(
        handle, n, (hipDoubleComplex*)A, lda, strideA, ipiv, strideP, info, batchCount);
#else
    return hipblasZgetrfStridedBatched(handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgetrs(
        handle, trans, n, nrhs, (hipComplex*)A, lda, ipiv, (hipComplex*)B, ldb, info);
#else
    return hipblasCgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgetrs(
        handle, trans, n, nrhs, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)B, ldb, info);
#else
    return hipblasZgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgetrsBatched(handle,
                                trans,
                                n,
                                nrhs,
                                (hipComplex* const*)A,
                                lda,
                                ipiv,
                                (hipComplex* const*)B,
                                ldb,
                                info,
                                batchCount);
#else
    return hipblasCgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgetrsBatched(handle,
                                trans,
                                n,
                                nrhs,
                                (hipDoubleComplex* const*)A,
                                lda,
                                ipiv,
                                (hipDoubleComplex* const*)B,
                                ldb,
                                info,
                                batchCount);
#else
    return hipblasZgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
#endif
}

// getrs_strided_batched
template <>
hipblasStatus_t hipblasGetrsStridedBatched<float>(hipblasHandle_t          handle,
                                                  const hipblasOperation_t trans,
                                                  const int                n,
                                                  const int                nrhs,
                                                  float*                   A,
                                                  const int                lda,
                                                  const hipblasStride      strideA,
                                                  const int*               ipiv,
                                                  const hipblasStride      strideP,
                                                  float*                   B,
                                                  const int                ldb,
                                                  const hipblasStride      strideB,
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
                                                   const hipblasStride      strideA,
                                                   const int*               ipiv,
                                                   const hipblasStride      strideP,
                                                   double*                  B,
                                                   const int                ldb,
                                                   const hipblasStride      strideB,
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
                                                           const hipblasStride      strideA,
                                                           const int*               ipiv,
                                                           const hipblasStride      strideP,
                                                           hipblasComplex*          B,
                                                           const int                ldb,
                                                           const hipblasStride      strideB,
                                                           int*                     info,
                                                           const int                batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetrsStridedBatched(handle,
                                       trans,
                                       n,
                                       nrhs,
                                       (hipComplex*)A,
                                       lda,
                                       strideA,
                                       ipiv,
                                       strideP,
                                       (hipComplex*)B,
                                       ldb,
                                       strideB,
                                       info,
                                       batchCount);
#else
    return hipblasCgetrsStridedBatched(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetrsStridedBatched<hipblasDoubleComplex>(hipblasHandle_t          handle,
                                                                 const hipblasOperation_t trans,
                                                                 const int                n,
                                                                 const int                nrhs,
                                                                 hipblasDoubleComplex*    A,
                                                                 const int                lda,
                                                                 const hipblasStride      strideA,
                                                                 const int*               ipiv,
                                                                 const hipblasStride      strideP,
                                                                 hipblasDoubleComplex*    B,
                                                                 const int                ldb,
                                                                 const hipblasStride      strideB,
                                                                 int*                     info,
                                                                 const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrsStridedBatched(handle,
                                       trans,
                                       n,
                                       nrhs,
                                       (hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       ipiv,
                                       strideP,
                                       (hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       info,
                                       batchCount);
#else
    return hipblasZgetrsStridedBatched(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
#endif
}

// getri_batched
template <>
hipblasStatus_t hipblasGetriBatched<float>(hipblasHandle_t handle,
                                           const int       n,
                                           float* const    A[],
                                           const int       lda,
                                           int*            ipiv,
                                           float* const    C[],
                                           const int       ldc,
                                           int*            info,
                                           const int       batchCount)
{
    return hipblasSgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetriBatched<double>(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            double* const   C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batchCount)
{
    return hipblasDgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetriBatched<hipblasComplex>(hipblasHandle_t       handle,
                                                    const int             n,
                                                    hipblasComplex* const A[],
                                                    const int             lda,
                                                    int*                  ipiv,
                                                    hipblasComplex* const C[],
                                                    const int             ldc,
                                                    int*                  info,
                                                    const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetriBatched(
        handle, n, (hipComplex* const*)A, lda, ipiv, (hipComplex* const*)C, ldc, info, batchCount);
#else
    return hipblasCgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetriBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
                                                          const int                   n,
                                                          hipblasDoubleComplex* const A[],
                                                          const int                   lda,
                                                          int*                        ipiv,
                                                          hipblasDoubleComplex* const C[],
                                                          const int                   ldc,
                                                          int*                        info,
                                                          const int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetriBatched(handle,
                                n,
                                (hipDoubleComplex* const*)A,
                                lda,
                                ipiv,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                info,
                                batchCount);
#else
    return hipblasZgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgeqrf(handle, m, n, (hipComplex*)A, lda, (hipComplex*)ipiv, info);
#else
    return hipblasCgeqrf(handle, m, n, A, lda, ipiv, info);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgeqrf(handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)ipiv, info);
#else
    return hipblasZgeqrf(handle, m, n, A, lda, ipiv, info);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasCgeqrfBatched(
        handle, m, n, (hipComplex* const*)A, lda, (hipComplex* const*)ipiv, info, batchCount);
#else
    return hipblasCgeqrfBatched(handle, m, n, A, lda, ipiv, info, batchCount);
#endif
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
#ifdef HIPBLAS_V2
    return hipblasZgeqrfBatched(handle,
                                m,
                                n,
                                (hipDoubleComplex* const*)A,
                                lda,
                                (hipDoubleComplex* const*)ipiv,
                                info,
                                batchCount);
#else
    return hipblasZgeqrfBatched(handle, m, n, A, lda, ipiv, info, batchCount);
#endif
}

// geqrf_strided_batched
template <>
hipblasStatus_t hipblasGeqrfStridedBatched<float>(hipblasHandle_t     handle,
                                                  const int           m,
                                                  const int           n,
                                                  float*              A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  float*              ipiv,
                                                  const hipblasStride strideP,
                                                  int*                info,
                                                  const int           batchCount)
{
    return hipblasSgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<double>(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride strideA,
                                                   double*             ipiv,
                                                   const hipblasStride strideP,
                                                   int*                info,
                                                   const int           batchCount)
{
    return hipblasDgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<hipblasComplex>(hipblasHandle_t     handle,
                                                           const int           m,
                                                           const int           n,
                                                           hipblasComplex*     A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           hipblasComplex*     ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgeqrfStridedBatched(
        handle, m, n, (hipComplex*)A, lda, strideA, (hipComplex*)ipiv, strideP, info, batchCount);
#else
    return hipblasCgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                                 const int             m,
                                                                 const int             n,
                                                                 hipblasDoubleComplex* A,
                                                                 const int             lda,
                                                                 const hipblasStride   strideA,
                                                                 hipblasDoubleComplex* ipiv,
                                                                 const hipblasStride   strideP,
                                                                 int*                  info,
                                                                 const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgeqrfStridedBatched(handle,
                                       m,
                                       n,
                                       (hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (hipDoubleComplex*)ipiv,
                                       strideP,
                                       info,
                                       batchCount);
#else
    return hipblasZgeqrfStridedBatched(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

// gels
template <>
hipblasStatus_t hipblasGels<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t trans,
                                   const int          m,
                                   const int          n,
                                   const int          nrhs,
                                   float*             A,
                                   const int          lda,
                                   float*             B,
                                   const int          ldb,
                                   int*               info,
                                   int*               deviceInfo)
{
    return hipblasSgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
}

template <>
hipblasStatus_t hipblasGels<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    double*            A,
                                    const int          lda,
                                    double*            B,
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo)
{
    return hipblasDgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
}

template <>
hipblasStatus_t hipblasGels<hipblasComplex>(hipblasHandle_t    handle,
                                            hipblasOperation_t trans,
                                            const int          m,
                                            const int          n,
                                            const int          nrhs,
                                            hipblasComplex*    A,
                                            const int          lda,
                                            hipblasComplex*    B,
                                            const int          ldb,
                                            int*               info,
                                            int*               deviceInfo)
{
#ifdef HIPBLAS_V2
    return hipblasCgels(
        handle, trans, m, n, nrhs, (hipComplex*)A, lda, (hipComplex*)B, ldb, info, deviceInfo);
#else
    return hipblasCgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
#endif
}

template <>
hipblasStatus_t hipblasGels<hipblasDoubleComplex>(hipblasHandle_t       handle,
                                                  hipblasOperation_t    trans,
                                                  const int             m,
                                                  const int             n,
                                                  const int             nrhs,
                                                  hipblasDoubleComplex* A,
                                                  const int             lda,
                                                  hipblasDoubleComplex* B,
                                                  const int             ldb,
                                                  int*                  info,
                                                  int*                  deviceInfo)
{
#ifdef HIPBLAS_V2
    return hipblasZgels(handle,
                        trans,
                        m,
                        n,
                        nrhs,
                        (hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)B,
                        ldb,
                        info,
                        deviceInfo);
#else
    return hipblasZgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
#endif
}

// gelsBatched
template <>
hipblasStatus_t hipblasGelsBatched<float>(hipblasHandle_t    handle,
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
                                          const int          batchCount)
{
    return hipblasSgelsBatched(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsBatched<double>(hipblasHandle_t    handle,
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
                                           const int          batchCount)
{
    return hipblasDgelsBatched(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsBatched<hipblasComplex>(hipblasHandle_t       handle,
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
                                                   const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgelsBatched(handle,
                               trans,
                               m,
                               n,
                               nrhs,
                               (hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)B,
                               ldb,
                               info,
                               deviceInfo,
                               batchCount);
#else
    return hipblasCgelsBatched(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGelsBatched<hipblasDoubleComplex>(hipblasHandle_t             handle,
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
                                                         const int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgelsBatched(handle,
                               trans,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)B,
                               ldb,
                               info,
                               deviceInfo,
                               batchCount);
#else
    return hipblasZgelsBatched(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
#endif
}

// gelsStridedBatched
template <>
hipblasStatus_t hipblasGelsStridedBatched<float>(hipblasHandle_t     handle,
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
                                                 const int           batchCount)
{
    return hipblasSgelsStridedBatched(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsStridedBatched<double>(hipblasHandle_t     handle,
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
                                                  const int           batchCount)
{
    return hipblasDgelsStridedBatched(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsStridedBatched<hipblasComplex>(hipblasHandle_t     handle,
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
                                                          const int           batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgelsStridedBatched(handle,
                                      trans,
                                      m,
                                      n,
                                      nrhs,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)B,
                                      ldb,
                                      strideB,
                                      info,
                                      deviceInfo,
                                      batchCount);
#else
    return hipblasCgelsStridedBatched(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGelsStridedBatched<hipblasDoubleComplex>(hipblasHandle_t       handle,
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
                                                                const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgelsStridedBatched(handle,
                                      trans,
                                      m,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      info,
                                      deviceInfo,
                                      batchCount);
#else
    return hipblasZgelsStridedBatched(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
#endif
}

#endif

/////////////
// FORTRAN //
/////////////
// /*
//  * ===========================================================================
//  *    level 3 BLAS
//  * ===========================================================================
//  */

// trtri
template <>
hipblasStatus_t hipblasTrtri<float, true>(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          hipblasDiagType_t diag,
                                          int               n,
                                          const float*      A,
                                          int               lda,
                                          float*            invA,
                                          int               ldinvA)
{
    return hipblasStrtriFortran(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
hipblasStatus_t hipblasTrtri<double, true>(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           hipblasDiagType_t diag,
                                           int               n,
                                           const double*     A,
                                           int               lda,
                                           double*           invA,
                                           int               ldinvA)
{
    return hipblasDtrtriFortran(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

template <>
hipblasStatus_t hipblasTrtri<hipblasComplex, true>(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasDiagType_t     diag,
                                                   int                   n,
                                                   const hipblasComplex* A,
                                                   int                   lda,
                                                   hipblasComplex*       invA,
                                                   int                   ldinvA)
{
#ifdef HIPBLAS_V2
    return hipblasCtrtriFortran(
        handle, uplo, diag, n, (const hipComplex*)A, lda, (hipComplex*)invA, ldinvA);
#else
    return hipblasCtrtriFortran(handle, uplo, diag, n, A, lda, invA, ldinvA);
#endif
}

template <>
hipblasStatus_t hipblasTrtri<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         hipblasDiagType_t           diag,
                                                         int                         n,
                                                         const hipblasDoubleComplex* A,
                                                         int                         lda,
                                                         hipblasDoubleComplex*       invA,
                                                         int                         ldinvA)
{
#ifdef HIPBLAS_V2
    return hipblasZtrtriFortran(
        handle, uplo, diag, n, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)invA, ldinvA);
#else
    return hipblasZtrtriFortran(handle, uplo, diag, n, A, lda, invA, ldinvA);
#endif
}

// trtri_batched
template <>
hipblasStatus_t hipblasTrtriBatched<float, true>(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasDiagType_t  diag,
                                                 int                n,
                                                 const float* const A[],
                                                 int                lda,
                                                 float*             invA[],
                                                 int                ldinvA,
                                                 int                batch_count)
{
    return hipblasStrtriBatchedFortran(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriBatched<double, true>(hipblasHandle_t     handle,
                                                  hipblasFillMode_t   uplo,
                                                  hipblasDiagType_t   diag,
                                                  int                 n,
                                                  const double* const A[],
                                                  int                 lda,
                                                  double*             invA[],
                                                  int                 ldinvA,
                                                  int                 batch_count)
{
    return hipblasDtrtriBatchedFortran(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriBatched<hipblasComplex, true>(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasDiagType_t           diag,
                                                          int                         n,
                                                          const hipblasComplex* const A[],
                                                          int                         lda,
                                                          hipblasComplex*             invA[],
                                                          int                         ldinvA,
                                                          int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrtriBatchedFortran(handle,
                                       uplo,
                                       diag,
                                       n,
                                       (const hipComplex* const*)A,
                                       lda,
                                       (hipComplex**)invA,
                                       ldinvA,
                                       batch_count);
#else
    return hipblasCtrtriBatchedFortran(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
#endif
}

template <>
hipblasStatus_t
    hipblasTrtriBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    hipblasDiagType_t                 diag,
                                                    int                               n,
                                                    const hipblasDoubleComplex* const A[],
                                                    int                               lda,
                                                    hipblasDoubleComplex*             invA[],
                                                    int                               ldinvA,
                                                    int                               batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrtriBatchedFortran(handle,
                                       uplo,
                                       diag,
                                       n,
                                       (const hipDoubleComplex* const*)A,
                                       lda,
                                       (hipDoubleComplex**)invA,
                                       ldinvA,
                                       batch_count);
#else
    return hipblasZtrtriBatchedFortran(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
#endif
}

// trtri_strided_batched
template <>
hipblasStatus_t hipblasTrtriStridedBatched<float, true>(hipblasHandle_t   handle,
                                                        hipblasFillMode_t uplo,
                                                        hipblasDiagType_t diag,
                                                        int               n,
                                                        const float*      A,
                                                        int               lda,
                                                        hipblasStride     stride_A,
                                                        float*            invA,
                                                        int               ldinvA,
                                                        hipblasStride     stride_invA,
                                                        int               batch_count)
{
    return hipblasStrtriStridedBatchedFortran(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriStridedBatched<double, true>(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         hipblasDiagType_t diag,
                                                         int               n,
                                                         const double*     A,
                                                         int               lda,
                                                         hipblasStride     stride_A,
                                                         double*           invA,
                                                         int               ldinvA,
                                                         hipblasStride     stride_invA,
                                                         int               batch_count)
{
    return hipblasDtrtriStridedBatchedFortran(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
}

template <>
hipblasStatus_t hipblasTrtriStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
                                                                 hipblasFillMode_t     uplo,
                                                                 hipblasDiagType_t     diag,
                                                                 int                   n,
                                                                 const hipblasComplex* A,
                                                                 int                   lda,
                                                                 hipblasStride         stride_A,
                                                                 hipblasComplex*       invA,
                                                                 int                   ldinvA,
                                                                 hipblasStride         stride_invA,
                                                                 int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrtriStridedBatchedFortran(handle,
                                              uplo,
                                              diag,
                                              n,
                                              (const hipComplex*)A,
                                              lda,
                                              stride_A,
                                              (hipComplex*)invA,
                                              ldinvA,
                                              stride_invA,
                                              batch_count);
#else
    return hipblasCtrtriStridedBatchedFortran(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
#endif
}

template <>
hipblasStatus_t
    hipblasTrtriStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           hipblasDiagType_t           diag,
                                                           int                         n,
                                                           const hipblasDoubleComplex* A,
                                                           int                         lda,
                                                           hipblasStride               stride_A,
                                                           hipblasDoubleComplex*       invA,
                                                           int                         ldinvA,
                                                           hipblasStride               stride_invA,
                                                           int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrtriStridedBatchedFortran(handle,
                                              uplo,
                                              diag,
                                              n,
                                              (const hipDoubleComplex*)A,
                                              lda,
                                              stride_A,
                                              (hipDoubleComplex*)invA,
                                              ldinvA,
                                              stride_invA,
                                              batch_count);
#else
    return hipblasZtrtriStridedBatchedFortran(
        handle, uplo, diag, n, A, lda, stride_A, invA, ldinvA, stride_invA, batch_count);
#endif
}

// dgmm
template <>
hipblasStatus_t hipblasDgmm<float, true>(hipblasHandle_t   handle,
                                         hipblasSideMode_t side,
                                         int               m,
                                         int               n,
                                         const float*      A,
                                         int               lda,
                                         const float*      x,
                                         int               incx,
                                         float*            C,
                                         int               ldc)
{
    return hipblasSdgmmFortran(handle, side, m, n, A, lda, x, incx, C, ldc);
}

template <>
hipblasStatus_t hipblasDgmm<double, true>(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          int               m,
                                          int               n,
                                          const double*     A,
                                          int               lda,
                                          const double*     x,
                                          int               incx,
                                          double*           C,
                                          int               ldc)
{
    return hipblasDdgmmFortran(handle, side, m, n, A, lda, x, incx, C, ldc);
}

template <>
hipblasStatus_t hipblasDgmm<hipblasComplex, true>(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  int                   m,
                                                  int                   n,
                                                  const hipblasComplex* A,
                                                  int                   lda,
                                                  const hipblasComplex* x,
                                                  int                   incx,
                                                  hipblasComplex*       C,
                                                  int                   ldc)
{
#ifdef HIPBLAS_V2
    return hipblasCdgmmFortran(handle,
                               side,
                               m,
                               n,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)x,
                               incx,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasCdgmmFortran(handle, side, m, n, A, lda, x, incx, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasDgmm<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                        hipblasSideMode_t           side,
                                                        int                         m,
                                                        int                         n,
                                                        const hipblasDoubleComplex* A,
                                                        int                         lda,
                                                        const hipblasDoubleComplex* x,
                                                        int                         incx,
                                                        hipblasDoubleComplex*       C,
                                                        int                         ldc)
{
#ifdef HIPBLAS_V2
    return hipblasZdgmmFortran(handle,
                               side,
                               m,
                               n,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)x,
                               incx,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZdgmmFortran(handle, side, m, n, A, lda, x, incx, C, ldc);
#endif
}

// dgmm_batched
template <>
hipblasStatus_t hipblasDgmmBatched<float, true>(hipblasHandle_t    handle,
                                                hipblasSideMode_t  side,
                                                int                m,
                                                int                n,
                                                const float* const A[],
                                                int                lda,
                                                const float* const x[],
                                                int                incx,
                                                float* const       C[],
                                                int                ldc,
                                                int                batch_count)
{
    return hipblasSdgmmBatchedFortran(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmBatched<double, true>(hipblasHandle_t     handle,
                                                 hipblasSideMode_t   side,
                                                 int                 m,
                                                 int                 n,
                                                 const double* const A[],
                                                 int                 lda,
                                                 const double* const x[],
                                                 int                 incx,
                                                 double* const       C[],
                                                 int                 ldc,
                                                 int                 batch_count)
{
    return hipblasDdgmmBatchedFortran(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmBatched<hipblasComplex, true>(hipblasHandle_t             handle,
                                                         hipblasSideMode_t           side,
                                                         int                         m,
                                                         int                         n,
                                                         const hipblasComplex* const A[],
                                                         int                         lda,
                                                         const hipblasComplex* const x[],
                                                         int                         incx,
                                                         hipblasComplex* const       C[],
                                                         int                         ldc,
                                                         int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCdgmmBatchedFortran(handle,
                                      side,
                                      m,
                                      n,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex* const*)x,
                                      incx,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batch_count);
#else
    return hipblasCdgmmBatchedFortran(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
#endif
}

template <>
hipblasStatus_t
    hipblasDgmmBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
                                                   hipblasSideMode_t                 side,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZdgmmBatchedFortran(handle,
                                      side,
                                      m,
                                      n,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex* const*)x,
                                      incx,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batch_count);
#else
    return hipblasZdgmmBatchedFortran(handle, side, m, n, A, lda, x, incx, C, ldc, batch_count);
#endif
}

// dgmm_strided_batched
template <>
hipblasStatus_t hipblasDgmmStridedBatched<float, true>(hipblasHandle_t   handle,
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
                                                       int               batch_count)
{
    return hipblasSdgmmStridedBatchedFortran(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmStridedBatched<double, true>(hipblasHandle_t   handle,
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
                                                        int               batch_count)
{
    return hipblasDdgmmStridedBatchedFortran(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
}

template <>
hipblasStatus_t hipblasDgmmStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCdgmmStridedBatchedFortran(handle,
                                             side,
                                             m,
                                             n,
                                             (const hipComplex*)A,
                                             lda,
                                             stride_A,
                                             (const hipComplex*)x,
                                             incx,
                                             stride_x,
                                             (hipComplex*)C,
                                             ldc,
                                             stride_C,
                                             batch_count);
#else
    return hipblasCdgmmStridedBatchedFortran(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
#endif
}

template <>
hipblasStatus_t hipblasDgmmStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t   handle,
                                                                      hipblasSideMode_t side,
                                                                      int               m,
                                                                      int               n,
                                                                      const hipblasDoubleComplex* A,
                                                                      int           lda,
                                                                      hipblasStride stride_A,
                                                                      const hipblasDoubleComplex* x,
                                                                      int           incx,
                                                                      hipblasStride stride_x,
                                                                      hipblasDoubleComplex* C,
                                                                      int                   ldc,
                                                                      hipblasStride stride_C,
                                                                      int           batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZdgmmStridedBatchedFortran(handle,
                                             side,
                                             m,
                                             n,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             stride_A,
                                             (const hipDoubleComplex*)x,
                                             incx,
                                             stride_x,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             stride_C,
                                             batch_count);
#else
    return hipblasZdgmmStridedBatchedFortran(
        handle, side, m, n, A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count);
#endif
}

// gemm
template <>
hipblasStatus_t hipblasGemm<hipblasHalf, true>(hipblasHandle_t    handle,
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
    return hipblasHgemmFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<float, true>(hipblasHandle_t    handle,
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
    return hipblasSgemmFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<double, true>(hipblasHandle_t    handle,
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
    return hipblasDgemmFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgemmFortran(handle,
                               transA,
                               transB,
                               m,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasCgemmFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasGemm<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZgemmFortran(handle,
                               transA,
                               transB,
                               m,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZgemmFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// gemm_batched
template <>
hipblasStatus_t hipblasGemmBatched<hipblasHalf, true>(hipblasHandle_t          handle,
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
    return hipblasHgemmBatchedFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<float, true>(hipblasHandle_t    handle,
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
    return hipblasSgemmBatchedFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<double, true>(hipblasHandle_t     handle,
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
    return hipblasDgemmBatchedFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgemmBatchedFortran(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex* const*)B,
                                      ldb,
                                      (const hipComplex*)beta,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batch_count);
#else
    return hipblasCgemmBatchedFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
#endif
}

template <>
hipblasStatus_t
    hipblasGemmBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
                                                   int                               batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZgemmBatchedFortran(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex* const*)B,
                                      ldb,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batch_count);
#else
    return hipblasZgemmBatchedFortran(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
#endif
}

// gemm_strided_batched
template <>
hipblasStatus_t hipblasGemmStridedBatched<hipblasHalf, true>(hipblasHandle_t    handle,
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

    return hipblasHgemmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasGemmStridedBatched<float, true>(hipblasHandle_t    handle,
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

    return hipblasSgemmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasGemmStridedBatched<double, true>(hipblasHandle_t    handle,
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

    return hipblasDgemmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasGemmStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgemmStridedBatchedFortran(handle,
                                             transA,
                                             transB,
                                             m,
                                             n,
                                             k,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             bsa,
                                             (const hipComplex*)B,
                                             ldb,
                                             bsb,
                                             (const hipComplex*)beta,
                                             (hipComplex*)C,
                                             ldc,
                                             bsc,
                                             batch_count);
#else
    return hipblasCgemmStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasGemmStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                          int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZgemmStridedBatchedFortran(handle,
                                             transA,
                                             transB,
                                             m,
                                             n,
                                             k,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             bsa,
                                             (const hipDoubleComplex*)B,
                                             ldb,
                                             bsb,
                                             (const hipDoubleComplex*)beta,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             bsc,
                                             batch_count);
#else
    return hipblasZgemmStridedBatchedFortran(handle,
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
#endif
}

// herk
template <>
hipblasStatus_t hipblasHerk<hipblasComplex, float, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCherkFortran(
        handle, uplo, transA, n, k, alpha, (const hipComplex*)A, lda, beta, (hipComplex*)C, ldc);
#else
    return hipblasCherkFortran(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasHerk<hipblasDoubleComplex, double, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZherkFortran(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               beta,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZherkFortran(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
}

// herk_batched
template <>
hipblasStatus_t hipblasHerkBatched<hipblasComplex, float, true>(hipblasHandle_t             handle,
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
                                                                int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCherkBatchedFortran(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      beta,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasCherkBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasHerkBatched<hipblasDoubleComplex, double, true>(hipblasHandle_t                   handle,
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
                                                           int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZherkBatchedFortran(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      beta,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasZherkBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
}

// herk_strided_batched
template <>
hipblasStatus_t hipblasHerkStridedBatched<hipblasComplex, float, true>(hipblasHandle_t       handle,
                                                                       hipblasFillMode_t     uplo,
                                                                       hipblasOperation_t    transA,
                                                                       int                   n,
                                                                       int                   k,
                                                                       const float*          alpha,
                                                                       const hipblasComplex* A,
                                                                       int                   lda,
                                                                       hipblasStride   strideA,
                                                                       const float*    beta,
                                                                       hipblasComplex* C,
                                                                       int             ldc,
                                                                       hipblasStride   strideC,
                                                                       int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCherkStridedBatchedFortran(handle,
                                             uplo,
                                             transA,
                                             n,
                                             k,
                                             alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             beta,
                                             (hipComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasCherkStridedBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasHerkStridedBatched<hipblasDoubleComplex, double, true>(hipblasHandle_t    handle,
                                                                  hipblasFillMode_t  uplo,
                                                                  hipblasOperation_t transA,
                                                                  int                n,
                                                                  int                k,
                                                                  const double*      alpha,
                                                                  const hipblasDoubleComplex* A,
                                                                  int                         lda,
                                                                  hipblasStride         strideA,
                                                                  const double*         beta,
                                                                  hipblasDoubleComplex* C,
                                                                  int                   ldc,
                                                                  hipblasStride         strideC,
                                                                  int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZherkStridedBatchedFortran(handle,
                                             uplo,
                                             transA,
                                             n,
                                             k,
                                             alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             beta,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasZherkStridedBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
}

// her2k
template <>
hipblasStatus_t hipblasHer2k<hipblasComplex, float, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCher2kFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex*)A,
                                lda,
                                (const hipComplex*)B,
                                ldb,
                                beta,
                                (hipComplex*)C,
                                ldc);
#else
    return hipblasCher2kFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasHer2k<hipblasDoubleComplex, double, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZher2kFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex*)A,
                                lda,
                                (const hipDoubleComplex*)B,
                                ldb,
                                beta,
                                (hipDoubleComplex*)C,
                                ldc);
#else
    return hipblasZher2kFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// her2k_batched
template <>
hipblasStatus_t hipblasHer2kBatched<hipblasComplex, float, true>(hipblasHandle_t             handle,
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
                                                                 int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCher2kBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex* const*)A,
                                       lda,
                                       (const hipComplex* const*)B,
                                       ldb,
                                       beta,
                                       (hipComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasCher2kBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasHer2kBatched<hipblasDoubleComplex, double, true>(hipblasHandle_t             handle,
                                                            hipblasFillMode_t           uplo,
                                                            hipblasOperation_t          transA,
                                                            int                         n,
                                                            int                         k,
                                                            const hipblasDoubleComplex* alpha,
                                                            const hipblasDoubleComplex* const A[],
                                                            int                               lda,
                                                            const hipblasDoubleComplex* const B[],
                                                            int                               ldb,
                                                            const double*                     beta,
                                                            hipblasDoubleComplex* const       C[],
                                                            int                               ldc,
                                                            int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZher2kBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex* const*)A,
                                       lda,
                                       (const hipDoubleComplex* const*)B,
                                       ldb,
                                       beta,
                                       (hipDoubleComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasZher2kBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

// her2k_strided_batched
template <>
hipblasStatus_t hipblasHer2kStridedBatched<hipblasComplex, float, true>(hipblasHandle_t    handle,
                                                                        hipblasFillMode_t  uplo,
                                                                        hipblasOperation_t transA,
                                                                        int                n,
                                                                        int                k,
                                                                        const hipblasComplex* alpha,
                                                                        const hipblasComplex* A,
                                                                        int                   lda,
                                                                        hipblasStride strideA,
                                                                        const hipblasComplex* B,
                                                                        int                   ldb,
                                                                        hipblasStride   strideB,
                                                                        const float*    beta,
                                                                        hipblasComplex* C,
                                                                        int             ldc,
                                                                        hipblasStride   strideC,
                                                                        int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCher2kStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipComplex*)alpha,
                                              (const hipComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipComplex*)B,
                                              ldb,
                                              strideB,
                                              beta,
                                              (hipComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasCher2kStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t hipblasHer2kStridedBatched<hipblasDoubleComplex, double, true>(
    hipblasHandle_t             handle,
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
    int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZher2kStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipDoubleComplex*)alpha,
                                              (const hipDoubleComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipDoubleComplex*)B,
                                              ldb,
                                              strideB,
                                              beta,
                                              (hipDoubleComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasZher2kStridedBatchedFortran(handle,
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
#endif
}

// herkx
template <>
hipblasStatus_t hipblasHerkx<hipblasComplex, float, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCherkxFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex*)A,
                                lda,
                                (const hipComplex*)B,
                                ldb,
                                beta,
                                (hipComplex*)C,
                                ldc);
#else
    return hipblasCherkxFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasHerkx<hipblasDoubleComplex, double, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZherkxFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex*)A,
                                lda,
                                (const hipDoubleComplex*)B,
                                ldb,
                                beta,
                                (hipDoubleComplex*)C,
                                ldc);
#else
    return hipblasZherkxFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// herkx_batched
template <>
hipblasStatus_t hipblasHerkxBatched<hipblasComplex, float, true>(hipblasHandle_t             handle,
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
                                                                 int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCherkxBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex* const*)A,
                                       lda,
                                       (const hipComplex* const*)B,
                                       ldb,
                                       beta,
                                       (hipComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasCherkxBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasHerkxBatched<hipblasDoubleComplex, double, true>(hipblasHandle_t             handle,
                                                            hipblasFillMode_t           uplo,
                                                            hipblasOperation_t          transA,
                                                            int                         n,
                                                            int                         k,
                                                            const hipblasDoubleComplex* alpha,
                                                            const hipblasDoubleComplex* const A[],
                                                            int                               lda,
                                                            const hipblasDoubleComplex* const B[],
                                                            int                               ldb,
                                                            const double*                     beta,
                                                            hipblasDoubleComplex* const       C[],
                                                            int                               ldc,
                                                            int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZherkxBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex* const*)A,
                                       lda,
                                       (const hipDoubleComplex* const*)B,
                                       ldb,
                                       beta,
                                       (hipDoubleComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasZherkxBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

// herkx_strided_batched
template <>
hipblasStatus_t hipblasHerkxStridedBatched<hipblasComplex, float, true>(hipblasHandle_t    handle,
                                                                        hipblasFillMode_t  uplo,
                                                                        hipblasOperation_t transA,
                                                                        int                n,
                                                                        int                k,
                                                                        const hipblasComplex* alpha,
                                                                        const hipblasComplex* A,
                                                                        int                   lda,
                                                                        hipblasStride strideA,
                                                                        const hipblasComplex* B,
                                                                        int                   ldb,
                                                                        hipblasStride   strideB,
                                                                        const float*    beta,
                                                                        hipblasComplex* C,
                                                                        int             ldc,
                                                                        hipblasStride   strideC,
                                                                        int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCherkxStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipComplex*)alpha,
                                              (const hipComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipComplex*)B,
                                              ldb,
                                              strideB,
                                              beta,
                                              (hipComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasCherkxStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t hipblasHerkxStridedBatched<hipblasDoubleComplex, double, true>(
    hipblasHandle_t             handle,
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
    int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZherkxStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipDoubleComplex*)alpha,
                                              (const hipDoubleComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipDoubleComplex*)B,
                                              ldb,
                                              strideB,
                                              beta,
                                              (hipDoubleComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasZherkxStridedBatchedFortran(handle,
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
#endif
}

// symm
template <>
hipblasStatus_t hipblasSymm<float, true>(hipblasHandle_t   handle,
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
    return hipblasSsymmFortran(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSymm<double, true>(hipblasHandle_t   handle,
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
    return hipblasDsymmFortran(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSymm<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsymmFortran(handle,
                               side,
                               uplo,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasCsymmFortran(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasSymm<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsymmFortran(handle,
                               side,
                               uplo,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZsymmFortran(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// symm_batched
template <>
hipblasStatus_t hipblasSymmBatched<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsymmBatchedFortran(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSymmBatched<double, true>(hipblasHandle_t     handle,
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
    return hipblasDsymmBatchedFortran(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSymmBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsymmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex* const*)B,
                                      ldb,
                                      (const hipComplex*)beta,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasCsymmBatchedFortran(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasSymmBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsymmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex* const*)B,
                                      ldb,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasZsymmBatchedFortran(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

// symm_strided_batched
template <>
hipblasStatus_t hipblasSymmStridedBatched<float, true>(hipblasHandle_t   handle,
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
                                                       int               batchCount)
{
    return hipblasSsymmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasSymmStridedBatched<double, true>(hipblasHandle_t   handle,
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
                                                        int               batchCount)
{
    return hipblasDsymmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasSymmStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsymmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             m,
                                             n,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipComplex*)B,
                                             ldb,
                                             strideB,
                                             (const hipComplex*)beta,
                                             (hipComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasCsymmStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasSymmStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsymmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             m,
                                             n,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipDoubleComplex*)B,
                                             ldb,
                                             strideB,
                                             (const hipDoubleComplex*)beta,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasZsymmStridedBatchedFortran(handle,
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
#endif
}

// syrk
template <>
hipblasStatus_t hipblasSyrk<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsyrkFortran(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrk<double, true>(hipblasHandle_t    handle,
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
    return hipblasDsyrkFortran(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrk<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkFortran(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)beta,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasCsyrkFortran(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasSyrk<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkFortran(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZsyrkFortran(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
#endif
}

// syrk_batched
template <>
hipblasStatus_t hipblasSyrkBatched<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsyrkBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkBatched<double, true>(hipblasHandle_t     handle,
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
    return hipblasDsyrkBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkBatchedFortran(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex*)beta,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasCsyrkBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasSyrkBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkBatchedFortran(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasZsyrkBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc, batchCount);
#endif
}

// syrk_strided_batched
template <>
hipblasStatus_t hipblasSyrkStridedBatched<float, true>(hipblasHandle_t    handle,
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
                                                       int                batchCount)
{
    return hipblasSsyrkStridedBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkStridedBatched<double, true>(hipblasHandle_t    handle,
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
                                                        int                batchCount)
{
    return hipblasDsyrkStridedBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsyrkStridedBatchedFortran(handle,
                                             uplo,
                                             transA,
                                             n,
                                             k,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipComplex*)beta,
                                             (hipComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasCsyrkStridedBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasSyrkStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsyrkStridedBatchedFortran(handle,
                                             uplo,
                                             transA,
                                             n,
                                             k,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipDoubleComplex*)beta,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasZsyrkStridedBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, strideA, beta, C, ldc, strideC, batchCount);
#endif
}

// syr2k
template <>
hipblasStatus_t hipblasSyr2k<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsyr2kFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyr2k<double, true>(hipblasHandle_t    handle,
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
    return hipblasDsyr2kFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyr2k<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsyr2kFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex*)A,
                                lda,
                                (const hipComplex*)B,
                                ldb,
                                (const hipComplex*)beta,
                                (hipComplex*)C,
                                ldc);
#else
    return hipblasCsyr2kFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasSyr2k<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsyr2kFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex*)A,
                                lda,
                                (const hipDoubleComplex*)B,
                                ldb,
                                (const hipDoubleComplex*)beta,
                                (hipDoubleComplex*)C,
                                ldc);
#else
    return hipblasZsyr2kFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// syr2k_batched
template <>
hipblasStatus_t hipblasSyr2kBatched<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsyr2kBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kBatched<double, true>(hipblasHandle_t     handle,
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
    return hipblasDsyr2kBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyr2kBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsyr2kBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex* const*)A,
                                       lda,
                                       (const hipComplex* const*)B,
                                       ldb,
                                       (const hipComplex*)beta,
                                       (hipComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasCsyr2kBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasSyr2kBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsyr2kBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex* const*)A,
                                       lda,
                                       (const hipDoubleComplex* const*)B,
                                       ldb,
                                       (const hipDoubleComplex*)beta,
                                       (hipDoubleComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasZsyr2kBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

// syr2k_strided_batched
template <>
hipblasStatus_t hipblasSyr2kStridedBatched<float, true>(hipblasHandle_t    handle,
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
                                                        int                batchCount)
{
    return hipblasSsyr2kStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasSyr2kStridedBatched<double, true>(hipblasHandle_t    handle,
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
                                                         int                batchCount)
{
    return hipblasDsyr2kStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasSyr2kStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                 int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsyr2kStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipComplex*)alpha,
                                              (const hipComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipComplex*)B,
                                              ldb,
                                              strideB,
                                              (const hipComplex*)beta,
                                              (hipComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasCsyr2kStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasSyr2kStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                           int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsyr2kStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipDoubleComplex*)alpha,
                                              (const hipDoubleComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipDoubleComplex*)B,
                                              ldb,
                                              strideB,
                                              (const hipDoubleComplex*)beta,
                                              (hipDoubleComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasZsyr2kStridedBatchedFortran(handle,
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
#endif
}

// syrkx
template <>
hipblasStatus_t hipblasSyrkx<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsyrkxFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrkx<double, true>(hipblasHandle_t    handle,
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
    return hipblasDsyrkxFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasSyrkx<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkxFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex*)A,
                                lda,
                                (const hipComplex*)B,
                                ldb,
                                (const hipComplex*)beta,
                                (hipComplex*)C,
                                ldc);
#else
    return hipblasCsyrkxFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasSyrkx<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkxFortran(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex*)A,
                                lda,
                                (const hipDoubleComplex*)B,
                                ldb,
                                (const hipDoubleComplex*)beta,
                                (hipDoubleComplex*)C,
                                ldc);
#else
    return hipblasZsyrkxFortran(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// syrkx_batched
template <>
hipblasStatus_t hipblasSyrkxBatched<float, true>(hipblasHandle_t    handle,
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
    return hipblasSsyrkxBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxBatched<double, true>(hipblasHandle_t     handle,
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
    return hipblasDsyrkxBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasSyrkxBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasCsyrkxBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex* const*)A,
                                       lda,
                                       (const hipComplex* const*)B,
                                       ldb,
                                       (const hipComplex*)beta,
                                       (hipComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasCsyrkxBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasSyrkxBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
#ifdef HIPBLAS_V2
    return hipblasZsyrkxBatchedFortran(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex* const*)A,
                                       lda,
                                       (const hipDoubleComplex* const*)B,
                                       ldb,
                                       (const hipDoubleComplex*)beta,
                                       (hipDoubleComplex* const*)C,
                                       ldc,
                                       batchCount);
#else
    return hipblasZsyrkxBatchedFortran(
        handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

// syrkx_strided_batched
template <>
hipblasStatus_t hipblasSyrkxStridedBatched<float, true>(hipblasHandle_t    handle,
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
                                                        int                batchCount)
{
    return hipblasSsyrkxStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasSyrkxStridedBatched<double, true>(hipblasHandle_t    handle,
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
                                                         int                batchCount)
{
    return hipblasDsyrkxStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasSyrkxStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                 int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCsyrkxStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipComplex*)alpha,
                                              (const hipComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipComplex*)B,
                                              ldb,
                                              strideB,
                                              (const hipComplex*)beta,
                                              (hipComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasCsyrkxStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasSyrkxStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                           int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZsyrkxStridedBatchedFortran(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              (const hipDoubleComplex*)alpha,
                                              (const hipDoubleComplex*)A,
                                              lda,
                                              strideA,
                                              (const hipDoubleComplex*)B,
                                              ldb,
                                              strideB,
                                              (const hipDoubleComplex*)beta,
                                              (hipDoubleComplex*)C,
                                              ldc,
                                              strideC,
                                              batchCount);
#else
    return hipblasZsyrkxStridedBatchedFortran(handle,
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
#endif
}

// hemm
template <>
hipblasStatus_t hipblasHemm<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasChemmFortran(handle,
                               side,
                               uplo,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasChemmFortran(handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasHemm<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZhemmFortran(handle,
                               side,
                               uplo,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZhemmFortran(handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// hemm_batched
template <>
hipblasStatus_t hipblasHemmBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasChemmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex* const*)B,
                                      ldb,
                                      (const hipComplex*)beta,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasChemmBatchedFortran(
        handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasHemmBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
#ifdef HIPBLAS_V2
    return hipblasZhemmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex* const*)B,
                                      ldb,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasZhemmBatchedFortran(
        handle, side, uplo, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif
}

// hemm_strided_batched
template <>
hipblasStatus_t hipblasHemmStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasChemmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             n,
                                             k,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipComplex*)B,
                                             ldb,
                                             strideB,
                                             (const hipComplex*)beta,
                                             (hipComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasChemmStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasHemmStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZhemmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             n,
                                             k,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipDoubleComplex*)B,
                                             ldb,
                                             strideB,
                                             (const hipDoubleComplex*)beta,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasZhemmStridedBatchedFortran(handle,
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
#endif
}

// trmm
template <>
hipblasStatus_t hipblasTrmm<float, true>(hipblasHandle_t    handle,
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
                                         int                ldc)
{
    return hipblasStrmmFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasTrmm<double, true>(hipblasHandle_t    handle,
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
                                          int                ldc)
{
    return hipblasDtrmmFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasTrmm<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                  int                   ldc)
{
#ifdef HIPBLAS_V2
    return hipblasCtrmmFortran(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)B,
                               ldb,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasCtrmmFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasTrmm<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                        int                         ldc)
{
#ifdef HIPBLAS_V2
    return hipblasZtrmmFortran(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZtrmmFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
}

// trmm_batched
template <>
hipblasStatus_t hipblasTrmmBatched<float, true>(hipblasHandle_t    handle,
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
                                                int                batchCount)
{
    return hipblasStrmmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasTrmmBatched<double, true>(hipblasHandle_t     handle,
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
                                                 int                 batchCount)
{
    return hipblasDtrmmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasTrmmBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
                                                         int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCtrmmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex* const*)B,
                                      ldb,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasCtrmmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasTrmmBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
                                                   int                               batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZtrmmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex* const*)B,
                                      ldb,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasZtrmmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc, batchCount);
#endif
}

// trmm_strided_batched
template <>
hipblasStatus_t hipblasTrmmStridedBatched<float, true>(hipblasHandle_t    handle,
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
                                                       int                batchCount)
{
    return hipblasStrmmStridedBatchedFortran(handle,
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
                                             C,
                                             ldc,
                                             strideC,
                                             batchCount);
}

template <>
hipblasStatus_t hipblasTrmmStridedBatched<double, true>(hipblasHandle_t    handle,
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
                                                        int                batchCount)
{
    return hipblasDtrmmStridedBatchedFortran(handle,
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
                                             C,
                                             ldc,
                                             strideC,
                                             batchCount);
}

template <>
hipblasStatus_t hipblasTrmmStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCtrmmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             m,
                                             n,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipComplex*)B,
                                             ldb,
                                             strideB,
                                             (hipComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasCtrmmStridedBatchedFortran(handle,
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
                                             C,
                                             ldc,
                                             strideC,
                                             batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasTrmmStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZtrmmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             m,
                                             n,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipDoubleComplex*)B,
                                             ldb,
                                             strideB,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasZtrmmStridedBatchedFortran(handle,
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
                                             C,
                                             ldc,
                                             strideC,
                                             batchCount);
#endif
}

// trsm
template <>
hipblasStatus_t hipblasTrsm<float, true>(hipblasHandle_t    handle,
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
    return hipblasStrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrsm<double, true>(hipblasHandle_t    handle,
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
    return hipblasDtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrsm<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCtrsmFortran(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (hipComplex*)B,
                               ldb);
#else
    return hipblasCtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
#endif
}

template <>
hipblasStatus_t hipblasTrsm<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZtrsmFortran(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb);
#else
    return hipblasZtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
#endif
}

// trsm_batched
template <>
hipblasStatus_t hipblasTrsmBatched<float, true>(hipblasHandle_t    handle,
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
                                                int                batch_count)
{
    return hipblasStrsmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

template <>
hipblasStatus_t hipblasTrsmBatched<double, true>(hipblasHandle_t     handle,
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
                                                 int                 batch_count)
{
    return hipblasDtrsmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

template <>
hipblasStatus_t hipblasTrsmBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
                                                         int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrsmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (hipComplex* const*)B,
                                      ldb,
                                      batch_count);
#else
    return hipblasCtrsmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
#endif
}

template <>
hipblasStatus_t
    hipblasTrsmBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
                                                   int                               batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrsmBatchedFortran(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (hipDoubleComplex* const*)B,
                                      ldb,
                                      batch_count);
#else
    return hipblasZtrsmBatchedFortran(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
#endif
}

// trsm_strided_batched
template <>
hipblasStatus_t hipblasTrsmStridedBatched<float, true>(hipblasHandle_t    handle,
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
                                                       int                batch_count)
{
    return hipblasStrsmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasTrsmStridedBatched<double, true>(hipblasHandle_t    handle,
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
                                                        int                batch_count)
{
    return hipblasDtrsmStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasTrsmStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                                int                   batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasCtrsmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             m,
                                             n,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             (hipComplex*)B,
                                             ldb,
                                             strideB,
                                             batch_count);
#else
    return hipblasCtrsmStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasTrsmStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
                                                          int                         batch_count)
{
#ifdef HIPBLAS_V2
    return hipblasZtrsmStridedBatchedFortran(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             m,
                                             n,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (hipDoubleComplex*)B,
                                             ldb,
                                             strideB,
                                             batch_count);
#else
    return hipblasZtrsmStridedBatchedFortran(handle,
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
#endif
}

// geam
template <>
hipblasStatus_t hipblasGeam<float, true>(hipblasHandle_t    handle,
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
    return hipblasSgeamFortran(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasGeam<double, true>(hipblasHandle_t    handle,
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
    return hipblasDgeamFortran(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasGeam<hipblasComplex, true>(hipblasHandle_t       handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgeamFortran(handle,
                               transA,
                               transB,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex*)A,
                               lda,
                               (const hipComplex*)beta,
                               (const hipComplex*)B,
                               ldb,
                               (hipComplex*)C,
                               ldc);
#else
    return hipblasCgeamFortran(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
}

template <>
hipblasStatus_t hipblasGeam<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasZgeamFortran(handle,
                               transA,
                               transB,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex*)A,
                               lda,
                               (const hipDoubleComplex*)beta,
                               (const hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)C,
                               ldc);
#else
    return hipblasZgeamFortran(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
}

// geam_batched
template <>
hipblasStatus_t hipblasGeamBatched<float, true>(hipblasHandle_t    handle,
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
    return hipblasSgeamBatchedFortran(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasGeamBatched<double, true>(hipblasHandle_t     handle,
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
    return hipblasDgeamBatchedFortran(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
}

template <>
hipblasStatus_t hipblasGeamBatched<hipblasComplex, true>(hipblasHandle_t             handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgeamBatchedFortran(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex* const*)A,
                                      lda,
                                      (const hipComplex*)beta,
                                      (const hipComplex* const*)B,
                                      ldb,
                                      (hipComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasCgeamBatchedFortran(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasGeamBatched<hipblasDoubleComplex, true>(hipblasHandle_t                   handle,
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
#ifdef HIPBLAS_V2
    return hipblasZgeamBatchedFortran(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex* const*)A,
                                      lda,
                                      (const hipDoubleComplex*)beta,
                                      (const hipDoubleComplex* const*)B,
                                      ldb,
                                      (hipDoubleComplex* const*)C,
                                      ldc,
                                      batchCount);
#else
    return hipblasZgeamBatchedFortran(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc, batchCount);
#endif
}

// geam_strided_batched
template <>
hipblasStatus_t hipblasGeamStridedBatched<float, true>(hipblasHandle_t    handle,
                                                       hipblasOperation_t transA,
                                                       hipblasOperation_t transB,
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
                                                       int                batchCount)
{
    return hipblasSgeamStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasGeamStridedBatched<double, true>(hipblasHandle_t    handle,
                                                        hipblasOperation_t transA,
                                                        hipblasOperation_t transB,
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
                                                        int                batchCount)
{
    return hipblasDgeamStridedBatchedFortran(handle,
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
hipblasStatus_t hipblasGeamStridedBatched<hipblasComplex, true>(hipblasHandle_t       handle,
                                                                hipblasOperation_t    transA,
                                                                hipblasOperation_t    transB,
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
                                                                int                   batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgeamStridedBatchedFortran(handle,
                                             transA,
                                             transB,
                                             m,
                                             n,
                                             (const hipComplex*)alpha,
                                             (const hipComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipComplex*)beta,
                                             (const hipComplex*)B,
                                             ldb,
                                             strideB,
                                             (hipComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasCgeamStridedBatchedFortran(handle,
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
#endif
}

template <>
hipblasStatus_t
    hipblasGeamStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                          hipblasOperation_t          transA,
                                                          hipblasOperation_t          transB,
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
                                                          int                         batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgeamStridedBatchedFortran(handle,
                                             transA,
                                             transB,
                                             m,
                                             n,
                                             (const hipDoubleComplex*)alpha,
                                             (const hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (const hipDoubleComplex*)beta,
                                             (const hipDoubleComplex*)B,
                                             ldb,
                                             strideB,
                                             (hipDoubleComplex*)C,
                                             ldc,
                                             strideC,
                                             batchCount);
#else
    return hipblasZgeamStridedBatchedFortran(handle,
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
#endif
}

#ifdef __HIP_PLATFORM_SOLVER__

// getrf
template <>
hipblasStatus_t hipblasGetrf<float, true>(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info)
{
    return hipblasSgetrfFortran(handle, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGetrf<double, true>(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info)
{
    return hipblasDgetrfFortran(handle, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGetrf<hipblasComplex, true>(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info)
{
#ifdef HIPBLAS_V2
    return hipblasCgetrfFortran(handle, n, (hipComplex*)A, lda, ipiv, info);
#else
    return hipblasCgetrfFortran(handle, n, A, lda, ipiv, info);
#endif
}

template <>
hipblasStatus_t hipblasGetrf<hipblasDoubleComplex, true>(hipblasHandle_t       handle,
                                                         const int             n,
                                                         hipblasDoubleComplex* A,
                                                         const int             lda,
                                                         int*                  ipiv,
                                                         int*                  info)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrfFortran(handle, n, (hipDoubleComplex*)A, lda, ipiv, info);
#else
    return hipblasZgetrfFortran(handle, n, A, lda, ipiv, info);
#endif
}

// getrf_batched
template <>
hipblasStatus_t hipblasGetrfBatched<float, true>(hipblasHandle_t handle,
                                                 const int       n,
                                                 float* const    A[],
                                                 const int       lda,
                                                 int*            ipiv,
                                                 int*            info,
                                                 const int       batchCount)
{
    return hipblasSgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfBatched<double, true>(hipblasHandle_t handle,
                                                  const int       n,
                                                  double* const   A[],
                                                  const int       lda,
                                                  int*            ipiv,
                                                  int*            info,
                                                  const int       batchCount)
{
    return hipblasDgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfBatched<hipblasComplex, true>(hipblasHandle_t       handle,
                                                          const int             n,
                                                          hipblasComplex* const A[],
                                                          const int             lda,
                                                          int*                  ipiv,
                                                          int*                  info,
                                                          const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetrfBatchedFortran(
        handle, n, (hipComplex* const*)A, lda, ipiv, info, batchCount);
#else
    return hipblasCgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetrfBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                                const int                   n,
                                                                hipblasDoubleComplex* const A[],
                                                                const int                   lda,
                                                                int*                        ipiv,
                                                                int*                        info,
                                                                const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrfBatchedFortran(
        handle, n, (hipDoubleComplex* const*)A, lda, ipiv, info, batchCount);
#else
    return hipblasZgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batchCount);
#endif
}

// getrf_strided_batched
template <>
hipblasStatus_t hipblasGetrfStridedBatched<float, true>(hipblasHandle_t     handle,
                                                        const int           n,
                                                        float*              A,
                                                        const int           lda,
                                                        const hipblasStride strideA,
                                                        int*                ipiv,
                                                        const hipblasStride strideP,
                                                        int*                info,
                                                        const int           batchCount)
{
    return hipblasSgetrfStridedBatchedFortran(
        handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<double, true>(hipblasHandle_t     handle,
                                                         const int           n,
                                                         double*             A,
                                                         const int           lda,
                                                         const hipblasStride strideA,
                                                         int*                ipiv,
                                                         const hipblasStride strideP,
                                                         int*                info,
                                                         const int           batchCount)
{
    return hipblasDgetrfStridedBatchedFortran(
        handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<hipblasComplex, true>(hipblasHandle_t     handle,
                                                                 const int           n,
                                                                 hipblasComplex*     A,
                                                                 const int           lda,
                                                                 const hipblasStride strideA,
                                                                 int*                ipiv,
                                                                 const hipblasStride strideP,
                                                                 int*                info,
                                                                 const int           batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetrfStridedBatchedFortran(
        handle, n, (hipComplex*)A, lda, strideA, ipiv, strideP, info, batchCount);
#else
    return hipblasCgetrfStridedBatchedFortran(
        handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetrfStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t       handle,
                                                                       const int             n,
                                                                       hipblasDoubleComplex* A,
                                                                       const int             lda,
                                                                       const hipblasStride strideA,
                                                                       int*                ipiv,
                                                                       const hipblasStride strideP,
                                                                       int*                info,
                                                                       const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrfStridedBatchedFortran(
        handle, n, (hipDoubleComplex*)A, lda, strideA, ipiv, strideP, info, batchCount);
#else
    return hipblasZgetrfStridedBatchedFortran(
        handle, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

// getrs
template <>
hipblasStatus_t hipblasGetrs<float, true>(hipblasHandle_t          handle,
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
    return hipblasSgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

template <>
hipblasStatus_t hipblasGetrs<double, true>(hipblasHandle_t          handle,
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
    return hipblasDgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

template <>
hipblasStatus_t hipblasGetrs<hipblasComplex, true>(hipblasHandle_t          handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgetrsFortran(
        handle, trans, n, nrhs, (hipComplex*)A, lda, ipiv, (hipComplex*)B, ldb, info);
#else
    return hipblasCgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
#endif
}

template <>
hipblasStatus_t hipblasGetrs<hipblasDoubleComplex, true>(hipblasHandle_t          handle,
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
#ifdef HIPBLAS_V2
    return hipblasZgetrsFortran(
        handle, trans, n, nrhs, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)B, ldb, info);
#else
    return hipblasZgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
#endif
}

// getrs_batched
template <>
hipblasStatus_t hipblasGetrsBatched<float, true>(hipblasHandle_t          handle,
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
    return hipblasSgetrsBatchedFortran(
        handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsBatched<double, true>(hipblasHandle_t          handle,
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
    return hipblasDgetrsBatchedFortran(
        handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsBatched<hipblasComplex, true>(hipblasHandle_t          handle,
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
#ifdef HIPBLAS_V2
    return hipblasCgetrsBatchedFortran(handle,
                                       trans,
                                       n,
                                       nrhs,
                                       (hipComplex* const*)A,
                                       lda,
                                       ipiv,
                                       (hipComplex* const*)B,
                                       ldb,
                                       info,
                                       batchCount);
#else
    return hipblasCgetrsBatchedFortran(
        handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetrsBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                                const hipblasOperation_t    trans,
                                                                const int                   n,
                                                                const int                   nrhs,
                                                                hipblasDoubleComplex* const A[],
                                                                const int                   lda,
                                                                const int*                  ipiv,
                                                                hipblasDoubleComplex* const B[],
                                                                const int                   ldb,
                                                                int*                        info,
                                                                const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrsBatchedFortran(handle,
                                       trans,
                                       n,
                                       nrhs,
                                       (hipDoubleComplex* const*)A,
                                       lda,
                                       ipiv,
                                       (hipDoubleComplex* const*)B,
                                       ldb,
                                       info,
                                       batchCount);
#else
    return hipblasZgetrsBatchedFortran(
        handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchCount);
#endif
}

// getrs_strided_batched
template <>
hipblasStatus_t hipblasGetrsStridedBatched<float, true>(hipblasHandle_t          handle,
                                                        const hipblasOperation_t trans,
                                                        const int                n,
                                                        const int                nrhs,
                                                        float*                   A,
                                                        const int                lda,
                                                        const hipblasStride      strideA,
                                                        const int*               ipiv,
                                                        const hipblasStride      strideP,
                                                        float*                   B,
                                                        const int                ldb,
                                                        const hipblasStride      strideB,
                                                        int*                     info,
                                                        const int                batchCount)
{
    return hipblasSgetrsStridedBatchedFortran(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsStridedBatched<double, true>(hipblasHandle_t          handle,
                                                         const hipblasOperation_t trans,
                                                         const int                n,
                                                         const int                nrhs,
                                                         double*                  A,
                                                         const int                lda,
                                                         const hipblasStride      strideA,
                                                         const int*               ipiv,
                                                         const hipblasStride      strideP,
                                                         double*                  B,
                                                         const int                ldb,
                                                         const hipblasStride      strideB,
                                                         int*                     info,
                                                         const int                batchCount)
{
    return hipblasDgetrsStridedBatchedFortran(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetrsStridedBatched<hipblasComplex, true>(hipblasHandle_t          handle,
                                                                 const hipblasOperation_t trans,
                                                                 const int                n,
                                                                 const int                nrhs,
                                                                 hipblasComplex*          A,
                                                                 const int                lda,
                                                                 const hipblasStride      strideA,
                                                                 const int*               ipiv,
                                                                 const hipblasStride      strideP,
                                                                 hipblasComplex*          B,
                                                                 const int                ldb,
                                                                 const hipblasStride      strideB,
                                                                 int*                     info,
                                                                 const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetrsStridedBatchedFortran(handle,
                                              trans,
                                              n,
                                              nrhs,
                                              (hipComplex*)A,
                                              lda,
                                              strideA,
                                              ipiv,
                                              strideP,
                                              (hipComplex*)B,
                                              ldb,
                                              strideB,
                                              info,
                                              batchCount);
#else
    return hipblasCgetrsStridedBatchedFortran(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
#endif
}

template <>
hipblasStatus_t
    hipblasGetrsStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t          handle,
                                                           const hipblasOperation_t trans,
                                                           const int                n,
                                                           const int                nrhs,
                                                           hipblasDoubleComplex*    A,
                                                           const int                lda,
                                                           const hipblasStride      strideA,
                                                           const int*               ipiv,
                                                           const hipblasStride      strideP,
                                                           hipblasDoubleComplex*    B,
                                                           const int                ldb,
                                                           const hipblasStride      strideB,
                                                           int*                     info,
                                                           const int                batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetrsStridedBatchedFortran(handle,
                                              trans,
                                              n,
                                              nrhs,
                                              (hipDoubleComplex*)A,
                                              lda,
                                              strideA,
                                              ipiv,
                                              strideP,
                                              (hipDoubleComplex*)B,
                                              ldb,
                                              strideB,
                                              info,
                                              batchCount);
#else
    return hipblasZgetrsStridedBatchedFortran(
        handle, trans, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batchCount);
#endif
}

// getri_batched
template <>
hipblasStatus_t hipblasGetriBatched<float, true>(hipblasHandle_t handle,
                                                 const int       n,
                                                 float* const    A[],
                                                 const int       lda,
                                                 int*            ipiv,
                                                 float* const    C[],
                                                 const int       ldc,
                                                 int*            info,
                                                 const int       batchCount)
{
    return hipblasSgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetriBatched<double, true>(hipblasHandle_t handle,
                                                  const int       n,
                                                  double* const   A[],
                                                  const int       lda,
                                                  int*            ipiv,
                                                  double* const   C[],
                                                  const int       ldc,
                                                  int*            info,
                                                  const int       batchCount)
{
    return hipblasDgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
}

template <>
hipblasStatus_t hipblasGetriBatched<hipblasComplex, true>(hipblasHandle_t       handle,
                                                          const int             n,
                                                          hipblasComplex* const A[],
                                                          const int             lda,
                                                          int*                  ipiv,
                                                          hipblasComplex* const C[],
                                                          const int             ldc,
                                                          int*                  info,
                                                          const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgetriBatchedFortran(
        handle, n, (hipComplex* const*)A, lda, ipiv, (hipComplex* const*)C, ldc, info, batchCount);
#else
    return hipblasCgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGetriBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                                const int                   n,
                                                                hipblasDoubleComplex* const A[],
                                                                const int                   lda,
                                                                int*                        ipiv,
                                                                hipblasDoubleComplex* const C[],
                                                                const int                   ldc,
                                                                int*                        info,
                                                                const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgetriBatchedFortran(handle,
                                       n,
                                       (hipDoubleComplex* const*)A,
                                       lda,
                                       ipiv,
                                       (hipDoubleComplex* const*)C,
                                       ldc,
                                       info,
                                       batchCount);
#else
    return hipblasZgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batchCount);
#endif
}

// geqrf
template <>
hipblasStatus_t hipblasGeqrf<float, true>(hipblasHandle_t handle,
                                          const int       m,
                                          const int       n,
                                          float*          A,
                                          const int       lda,
                                          float*          ipiv,
                                          int*            info)
{
    return hipblasSgeqrfFortran(handle, m, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGeqrf<double, true>(hipblasHandle_t handle,
                                           const int       m,
                                           const int       n,
                                           double*         A,
                                           const int       lda,
                                           double*         ipiv,
                                           int*            info)
{
    return hipblasDgeqrfFortran(handle, m, n, A, lda, ipiv, info);
}

template <>
hipblasStatus_t hipblasGeqrf<hipblasComplex, true>(hipblasHandle_t handle,
                                                   const int       m,
                                                   const int       n,
                                                   hipblasComplex* A,
                                                   const int       lda,
                                                   hipblasComplex* ipiv,
                                                   int*            info)
{
#ifdef HIPBLAS_V2
    return hipblasCgeqrfFortran(handle, m, n, (hipComplex*)A, lda, (hipComplex*)ipiv, info);
#else
    return hipblasCgeqrfFortran(handle, m, n, A, lda, ipiv, info);
#endif
}

template <>
hipblasStatus_t hipblasGeqrf<hipblasDoubleComplex, true>(hipblasHandle_t       handle,
                                                         const int             m,
                                                         const int             n,
                                                         hipblasDoubleComplex* A,
                                                         const int             lda,
                                                         hipblasDoubleComplex* ipiv,
                                                         int*                  info)
{
#ifdef HIPBLAS_V2
    return hipblasZgeqrfFortran(
        handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)ipiv, info);
#else
    return hipblasZgeqrfFortran(handle, m, n, A, lda, ipiv, info);
#endif
}

// geqrf_batched
template <>
hipblasStatus_t hipblasGeqrfBatched<float, true>(hipblasHandle_t handle,
                                                 const int       m,
                                                 const int       n,
                                                 float* const    A[],
                                                 const int       lda,
                                                 float* const    ipiv[],
                                                 int*            info,
                                                 const int       batchCount)
{
    return hipblasSgeqrfBatchedFortran(handle, m, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfBatched<double, true>(hipblasHandle_t handle,
                                                  const int       m,
                                                  const int       n,
                                                  double* const   A[],
                                                  const int       lda,
                                                  double* const   ipiv[],
                                                  int*            info,
                                                  const int       batchCount)
{
    return hipblasDgeqrfBatchedFortran(handle, m, n, A, lda, ipiv, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfBatched<hipblasComplex, true>(hipblasHandle_t       handle,
                                                          const int             m,
                                                          const int             n,
                                                          hipblasComplex* const A[],
                                                          const int             lda,
                                                          hipblasComplex* const ipiv[],
                                                          int*                  info,
                                                          const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgeqrfBatchedFortran(
        handle, m, n, (hipComplex* const*)A, lda, (hipComplex* const*)ipiv, info, batchCount);
#else
    return hipblasCgeqrfBatchedFortran(handle, m, n, A, lda, ipiv, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGeqrfBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                                const int                   m,
                                                                const int                   n,
                                                                hipblasDoubleComplex* const A[],
                                                                const int                   lda,
                                                                hipblasDoubleComplex* const ipiv[],
                                                                int*                        info,
                                                                const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgeqrfBatchedFortran(handle,
                                       m,
                                       n,
                                       (hipDoubleComplex* const*)A,
                                       lda,
                                       (hipDoubleComplex* const*)ipiv,
                                       info,
                                       batchCount);
#else
    return hipblasZgeqrfBatchedFortran(handle, m, n, A, lda, ipiv, info, batchCount);
#endif
}

// geqrf_strided_batched
template <>
hipblasStatus_t hipblasGeqrfStridedBatched<float, true>(hipblasHandle_t     handle,
                                                        const int           m,
                                                        const int           n,
                                                        float*              A,
                                                        const int           lda,
                                                        const hipblasStride strideA,
                                                        float*              ipiv,
                                                        const hipblasStride strideP,
                                                        int*                info,
                                                        const int           batchCount)
{
    return hipblasSgeqrfStridedBatchedFortran(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<double, true>(hipblasHandle_t     handle,
                                                         const int           m,
                                                         const int           n,
                                                         double*             A,
                                                         const int           lda,
                                                         const hipblasStride strideA,
                                                         double*             ipiv,
                                                         const hipblasStride strideP,
                                                         int*                info,
                                                         const int           batchCount)
{
    return hipblasDgeqrfStridedBatchedFortran(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<hipblasComplex, true>(hipblasHandle_t     handle,
                                                                 const int           m,
                                                                 const int           n,
                                                                 hipblasComplex*     A,
                                                                 const int           lda,
                                                                 const hipblasStride strideA,
                                                                 hipblasComplex*     ipiv,
                                                                 const hipblasStride strideP,
                                                                 int*                info,
                                                                 const int           batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgeqrfStridedBatchedFortran(
        handle, m, n, (hipComplex*)A, lda, strideA, (hipComplex*)ipiv, strideP, info, batchCount);
#else
    return hipblasCgeqrfStridedBatchedFortran(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGeqrfStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t       handle,
                                                                       const int             m,
                                                                       const int             n,
                                                                       hipblasDoubleComplex* A,
                                                                       const int             lda,
                                                                       const hipblasStride strideA,
                                                                       hipblasDoubleComplex* ipiv,
                                                                       const hipblasStride strideP,
                                                                       int*                info,
                                                                       const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgeqrfStridedBatchedFortran(handle,
                                              m,
                                              n,
                                              (hipDoubleComplex*)A,
                                              lda,
                                              strideA,
                                              (hipDoubleComplex*)ipiv,
                                              strideP,
                                              info,
                                              batchCount);
#else
    return hipblasZgeqrfStridedBatchedFortran(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batchCount);
#endif
}

// gels
template <>
hipblasStatus_t hipblasGels<float, true>(hipblasHandle_t    handle,
                                         hipblasOperation_t trans,
                                         const int          m,
                                         const int          n,
                                         const int          nrhs,
                                         float*             A,
                                         const int          lda,
                                         float*             B,
                                         const int          ldb,
                                         int*               info,
                                         int*               deviceInfo)
{
    return hipblasSgelsFortran(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
}

template <>
hipblasStatus_t hipblasGels<double, true>(hipblasHandle_t    handle,
                                          hipblasOperation_t trans,
                                          const int          m,
                                          const int          n,
                                          const int          nrhs,
                                          double*            A,
                                          const int          lda,
                                          double*            B,
                                          const int          ldb,
                                          int*               info,
                                          int*               deviceInfo)
{
    return hipblasDgelsFortran(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
}

template <>
hipblasStatus_t hipblasGels<hipblasComplex, true>(hipblasHandle_t    handle,
                                                  hipblasOperation_t trans,
                                                  const int          m,
                                                  const int          n,
                                                  const int          nrhs,
                                                  hipblasComplex*    A,
                                                  const int          lda,
                                                  hipblasComplex*    B,
                                                  const int          ldb,
                                                  int*               info,
                                                  int*               deviceInfo)
{
#ifdef HIPBLAS_V2
    return hipblasCgelsFortran(
        handle, trans, m, n, nrhs, (hipComplex*)A, lda, (hipComplex*)B, ldb, info, deviceInfo);
#else
    return hipblasCgelsFortran(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
#endif
}

template <>
hipblasStatus_t hipblasGels<hipblasDoubleComplex, true>(hipblasHandle_t       handle,
                                                        hipblasOperation_t    trans,
                                                        const int             m,
                                                        const int             n,
                                                        const int             nrhs,
                                                        hipblasDoubleComplex* A,
                                                        const int             lda,
                                                        hipblasDoubleComplex* B,
                                                        const int             ldb,
                                                        int*                  info,
                                                        int*                  deviceInfo)
{
#ifdef HIPBLAS_V2
    return hipblasZgelsFortran(handle,
                               trans,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               info,
                               deviceInfo);
#else
    return hipblasZgelsFortran(handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo);
#endif
}

// gelsBatched
template <>
hipblasStatus_t hipblasGelsBatched<float, true>(hipblasHandle_t    handle,
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
                                                const int          batchCount)
{
    return hipblasSgelsBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsBatched<double, true>(hipblasHandle_t    handle,
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
                                                 const int          batchCount)
{
    return hipblasDgelsBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsBatched<hipblasComplex, true>(hipblasHandle_t       handle,
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
                                                         const int             batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgelsBatchedFortran(handle,
                                      trans,
                                      m,
                                      n,
                                      nrhs,
                                      (hipComplex* const*)A,
                                      lda,
                                      (hipComplex* const*)B,
                                      ldb,
                                      info,
                                      deviceInfo,
                                      batchCount);
#else
    return hipblasCgelsBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGelsBatched<hipblasDoubleComplex, true>(hipblasHandle_t             handle,
                                                               hipblasOperation_t          trans,
                                                               const int                   m,
                                                               const int                   n,
                                                               const int                   nrhs,
                                                               hipblasDoubleComplex* const A[],
                                                               const int                   lda,
                                                               hipblasDoubleComplex* const B[],
                                                               const int                   ldb,
                                                               int*                        info,
                                                               int*      deviceInfo,
                                                               const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgelsBatchedFortran(handle,
                                      trans,
                                      m,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex* const*)A,
                                      lda,
                                      (hipDoubleComplex* const*)B,
                                      ldb,
                                      info,
                                      deviceInfo,
                                      batchCount);
#else
    return hipblasZgelsBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, B, ldb, info, deviceInfo, batchCount);
#endif
}

// gelsStridedBatched
template <>
hipblasStatus_t hipblasGelsStridedBatched<float, true>(hipblasHandle_t     handle,
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
                                                       const int           batchCount)
{
    return hipblasSgelsStridedBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsStridedBatched<double, true>(hipblasHandle_t     handle,
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
                                                        const int           batchCount)
{
    return hipblasDgelsStridedBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
}

template <>
hipblasStatus_t hipblasGelsStridedBatched<hipblasComplex, true>(hipblasHandle_t     handle,
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
                                                                const int           batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasCgelsStridedBatchedFortran(handle,
                                             trans,
                                             m,
                                             n,
                                             nrhs,
                                             (hipComplex*)A,
                                             lda,
                                             strideA,
                                             (hipComplex*)B,
                                             ldb,
                                             strideB,
                                             info,
                                             deviceInfo,
                                             batchCount);
#else
    return hipblasCgelsStridedBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
#endif
}

template <>
hipblasStatus_t hipblasGelsStridedBatched<hipblasDoubleComplex, true>(hipblasHandle_t       handle,
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
                                                                      int*      deviceInfo,
                                                                      const int batchCount)
{
#ifdef HIPBLAS_V2
    return hipblasZgelsStridedBatchedFortran(handle,
                                             trans,
                                             m,
                                             n,
                                             nrhs,
                                             (hipDoubleComplex*)A,
                                             lda,
                                             strideA,
                                             (hipDoubleComplex*)B,
                                             ldb,
                                             strideB,
                                             info,
                                             deviceInfo,
                                             batchCount);
#else
    return hipblasZgelsStridedBatchedFortran(
        handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, strideB, info, deviceInfo, batchCount);
#endif
}

#endif
