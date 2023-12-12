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
#ifndef _HIPBLAS_HPP_
#define _HIPBLAS_HPP_

/* library headers */
#include "hipblas.h"

#ifndef WIN32
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif

#if not defined(__clang_major__)
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define MAP2CF(...) GET_MACRO(__VA_ARGS__, MAP2CF5, MAP2CF4, MAP2CF3, dum2, dum1)(__VA_ARGS__)
// dual API C and FORTRAN
#define MAP2CF_D64(...) \
    GET_MACRO(__VA_ARGS__, MAP2DCF5, MAP2DCF4, MAP2DCF3, dum2, dum1)(__VA_ARGS__)

#if !defined(HIPBLAS_V2) \
    && !defined(         \
        WIN32) // HIPBLAS_V2 doesn't have fortran tests during transition period, WIN doesn't have fortran tests
#define MAP2CF3(FN, A, PFN)  \
    template <>              \
    auto FN<A, false> = PFN; \
    template <>              \
    auto FN<A, true> = PFN##Fortran
#define MAP2CF4(FN, A, B, PFN)  \
    template <>                 \
    auto FN<A, B, false> = PFN; \
    template <>                 \
    auto FN<A, B, true> = PFN##Fortran
#define MAP2CF5(FN, A, B, C, PFN)  \
    template <>                    \
    auto FN<A, B, C, false> = PFN; \
    template <>                    \
    auto FN<A, B, C, true> = PFN##Fortran
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)           \
    template <>                        \
    auto FN<A, false> = PFN;           \
    template <>                        \
    auto FN<A, true> = PFN##Fortran;   \
    template <>                        \
    auto FN##_64<A, false> = PFN##_64; \
    template <>                        \
    auto FN##_64<A, true> = PFN##_64Fortran
#define MAP2DCF4(FN, A, B, PFN)           \
    template <>                           \
    auto FN<A, B, false> = PFN;           \
    template <>                           \
    auto FN<A, B, true> = PFN##Fortran;   \
    template <>                           \
    auto FN##_64<A, B, false> = PFN##_64; \
    template <>                           \
    auto FN##_64<A, B, true> = PFN##_64Fortran
#define MAP2DCF5(FN, A, B, C, PFN)           \
    template <>                              \
    auto FN<A, B, C, false> = PFN;           \
    template <>                              \
    auto FN<A, B, C, true> = PFN##Fortran;   \
    template <>                              \
    auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                              \
    auto FN##_64<A, B, C, true> = PFN##_64Fortran
#else
// mapping fortran and C to C API
#define MAP2CF3(FN, A, PFN)  \
    template <>              \
    auto FN<A, false> = PFN; \
    template <>              \
    auto FN<A, true> = PFN
#define MAP2CF4(FN, A, B, PFN)  \
    template <>                 \
    auto FN<A, B, false> = PFN; \
    template <>                 \
    auto FN<A, B, true> = PFN
#define MAP2CF5(FN, A, B, C, PFN)  \
    template <>                    \
    auto FN<A, B, C, false> = PFN; \
    template <>                    \
    auto FN<A, B, C, true> = PFN
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)           \
    template <>                        \
    auto FN<A, false> = PFN;           \
    template <>                        \
    auto FN<A, true> = PFN;            \
    template <>                        \
    auto FN##_64<A, false> = PFN##_64; \
    template <>                        \
    auto FN##_64<A, true> = PFN##_64
#define MAP2DCF4(FN, A, B, PFN)           \
    template <>                           \
    auto FN<A, B, false> = PFN;           \
    template <>                           \
    auto FN<A, B, true> = PFN;            \
    template <>                           \
    auto FN##_64<A, B, false> = PFN##_64; \
    template <>                           \
    auto FN##_64<A, B, true> = PFN##_64
#define MAP2DCF5(FN, A, B, C, PFN)           \
    template <>                              \
    auto FN<A, B, C, false> = PFN;           \
    template <>                              \
    auto FN<A, B, C, true> = PFN;            \
    template <>                              \
    auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                              \
    auto FN##_64<A, B, C, true> = PFN##_64
#endif

#ifndef HIPBLAS_V2
#define MAP2CF_D64_V2(...) MAP2CF_D64(__VA_ARGS__)
#define MAP2CF_V2(...) MAP2CF(__VA_ARGS__)
#else
#define MAP2CF_D64_V2(...) MAP2CF_D64(__VA_ARGS__##Cast)
#define MAP2CF_V2(...) MAP2CF(__VA_ARGS__##Cast)
#endif

// Need these temporarily during transition period between hipblasComplex -> hipComplex
hipblasStatus_t hipblasCscalCast(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx);
hipblasStatus_t hipblasCsscalCast(
    hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx);
hipblasStatus_t hipblasZscalCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);
hipblasStatus_t hipblasZdscalCast(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx);
hipblasStatus_t hipblasCscalBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        const hipblasComplex* alpha,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        int                   batch_count);
hipblasStatus_t hipblasCsscalBatchedCast(hipblasHandle_t       handle,
                                         int                   n,
                                         const float*          alpha,
                                         hipblasComplex* const x[],
                                         int                   incx,
                                         int                   batch_count);
hipblasStatus_t hipblasZscalBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasDoubleComplex* alpha,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        int                         batch_count);
hipblasStatus_t hipblasZdscalBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const double*               alpha,
                                         hipblasDoubleComplex* const x[],
                                         int                         incx,
                                         int                         batch_count);
hipblasStatus_t hipblasCscalStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batch_count);
hipblasStatus_t hipblasCsscalStridedBatchedCast(hipblasHandle_t handle,
                                                int             n,
                                                const float*    alpha,
                                                hipblasComplex* x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                int             batch_count);
hipblasStatus_t hipblasZscalStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batch_count);
hipblasStatus_t hipblasZdscalStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const double*         alpha,
                                                hipblasDoubleComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count);

// copy
hipblasStatus_t hipblasCcopyCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);
hipblasStatus_t hipblasZcopyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy);
hipblasStatus_t hipblasCcopyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count);
hipblasStatus_t hipblasZcopyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count);
hipblasStatus_t hipblasCcopyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count);
hipblasStatus_t hipblasZcopyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count);

// axpy
hipblasStatus_t hipblasCaxpyCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 hipblasComplex*       y,
                                 int                   incy);

hipblasStatus_t hipblasZaxpyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy);

hipblasStatus_t hipblasCaxpyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count);

hipblasStatus_t hipblasZaxpyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count);

hipblasStatus_t hipblasCaxpyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count);

hipblasStatus_t hipblasZaxpyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count);

// swap
hipblasStatus_t hipblasCswapCast(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy);

hipblasStatus_t hipblasZswapCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy);

hipblasStatus_t hipblasCswapBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        int                   batch_count);

hipblasStatus_t hipblasZswapBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        int                         batch_count);

hipblasStatus_t hipblasCswapStridedBatchedCast(hipblasHandle_t handle,
                                               int             n,
                                               hipblasComplex* x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               hipblasComplex* y,
                                               int             incy,
                                               hipblasStride   stridey,
                                               int             batch_count);

hipblasStatus_t hipblasZswapStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               hipblasDoubleComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasDoubleComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count);

// dot
hipblasStatus_t hipblasCdotuCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result);

hipblasStatus_t hipblasCdotcCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result);

hipblasStatus_t hipblasZdotuCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result);

hipblasStatus_t hipblasZdotcCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result);

hipblasStatus_t hipblasCdotuBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result);

hipblasStatus_t hipblasCdotcBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result);

hipblasStatus_t hipblasZdotuBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result);

hipblasStatus_t hipblasZdotcBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result);

hipblasStatus_t hipblasCdotuStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result);

hipblasStatus_t hipblasCdotcStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result);

hipblasStatus_t hipblasZdotuStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result);

hipblasStatus_t hipblasZdotcStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result);

// asum
hipblasStatus_t hipblasScasumCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

hipblasStatus_t hipblasDzasumCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);

hipblasStatus_t hipblasScasumBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result);

hipblasStatus_t hipblasDzasumBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result);

hipblasStatus_t hipblasScasumStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result);

hipblasStatus_t hipblasDzasumStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result);

// nrm2
hipblasStatus_t hipblasScnrm2Cast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

hipblasStatus_t hipblasDznrm2Cast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);

hipblasStatus_t hipblasScnrm2BatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result);

hipblasStatus_t hipblasDznrm2BatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result);

hipblasStatus_t hipblasScnrm2StridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result);

hipblasStatus_t hipblasDznrm2StridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result);

// rot
hipblasStatus_t hipblasCrotCast(hipblasHandle_t       handle,
                                int                   n,
                                hipblasComplex*       x,
                                int                   incx,
                                hipblasComplex*       y,
                                int                   incy,
                                const float*          c,
                                const hipblasComplex* s);

hipblasStatus_t hipblasCsrotCast(hipblasHandle_t handle,
                                 int             n,
                                 hipblasComplex* x,
                                 int             incx,
                                 hipblasComplex* y,
                                 int             incy,
                                 const float*    c,
                                 const float*    s);

hipblasStatus_t hipblasZrotCast(hipblasHandle_t             handle,
                                int                         n,
                                hipblasDoubleComplex*       x,
                                int                         incx,
                                hipblasDoubleComplex*       y,
                                int                         incy,
                                const double*               c,
                                const hipblasDoubleComplex* s);

hipblasStatus_t hipblasZdrotCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy,
                                 const double*         c,
                                 const double*         s);

hipblasStatus_t hipblasCrotBatchedCast(hipblasHandle_t       handle,
                                       int                   n,
                                       hipblasComplex* const x[],
                                       int                   incx,
                                       hipblasComplex* const y[],
                                       int                   incy,
                                       const float*          c,
                                       const hipblasComplex* s,
                                       int                   batch_count);

hipblasStatus_t hipblasCsrotBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        const float*          c,
                                        const float*          s,
                                        int                   batch_count);

hipblasStatus_t hipblasZrotBatchedCast(hipblasHandle_t             handle,
                                       int                         n,
                                       hipblasDoubleComplex* const x[],
                                       int                         incx,
                                       hipblasDoubleComplex* const y[],
                                       int                         incy,
                                       const double*               c,
                                       const hipblasDoubleComplex* s,
                                       int                         batch_count);

hipblasStatus_t hipblasZdrotBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        const double*               c,
                                        const double*               s,
                                        int                         batch_count);

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
                                              int                   batch_count);

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
                                               int             batch_count);

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
                                              int                         batch_count);

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
                                               int                   batch_count);

// rotg
hipblasStatus_t hipblasCrotgCast(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);

hipblasStatus_t hipblasZrotgCast(hipblasHandle_t       handle,
                                 hipblasDoubleComplex* a,
                                 hipblasDoubleComplex* b,
                                 double*               c,
                                 hipblasDoubleComplex* s);

hipblasStatus_t hipblasCrotgBatchedCast(hipblasHandle_t       handle,
                                        hipblasComplex* const a[],
                                        hipblasComplex* const b[],
                                        float* const          c[],
                                        hipblasComplex* const s[],
                                        int                   batch_count);

hipblasStatus_t hipblasZrotgBatchedCast(hipblasHandle_t             handle,
                                        hipblasDoubleComplex* const a[],
                                        hipblasDoubleComplex* const b[],
                                        double* const               c[],
                                        hipblasDoubleComplex* const s[],
                                        int                         batch_count);

hipblasStatus_t hipblasCrotgStridedBatchedCast(hipblasHandle_t handle,
                                               hipblasComplex* a,
                                               hipblasStride   stridea,
                                               hipblasComplex* b,
                                               hipblasStride   strideb,
                                               float*          c,
                                               hipblasStride   stridec,
                                               hipblasComplex* s,
                                               hipblasStride   strides,
                                               int             batch_count);

hipblasStatus_t hipblasZrotgStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasDoubleComplex* a,
                                               hipblasStride         stridea,
                                               hipblasDoubleComplex* b,
                                               hipblasStride         strideb,
                                               double*               c,
                                               hipblasStride         stridec,
                                               hipblasDoubleComplex* s,
                                               hipblasStride         strides,
                                               int                   batch_count);

// rotm, rotmg - no complex versions

// amax
hipblasStatus_t hipblasIcamaxCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzamaxCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);

hipblasStatus_t hipblasIcamaxBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result);

hipblasStatus_t hipblasIzamaxBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result);

hipblasStatus_t hipblasIcamaxStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result);

hipblasStatus_t hipblasIzamaxStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result);

// amin
hipblasStatus_t hipblasIcaminCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzaminCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);

hipblasStatus_t hipblasIcaminBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result);

hipblasStatus_t hipblasIzaminBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result);

hipblasStatus_t hipblasIcaminStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result);

hipblasStatus_t hipblasIzaminStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result);

namespace
{
    // Scal
    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScal)(hipblasHandle_t handle, int n, const U* alpha, T* x, int incx);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalBatched)(
        hipblasHandle_t handle, int n, const U* alpha, T* const x[], int incx, int batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const U*        alpha,
                                                 T*              x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 int             batch_count);

    MAP2CF(hipblasScal, float, float, hipblasSscal);
    MAP2CF(hipblasScal, double, double, hipblasDscal);
    MAP2CF_V2(hipblasScal, hipblasComplex, hipblasComplex, hipblasCscal);
    MAP2CF_V2(hipblasScal, hipblasDoubleComplex, hipblasDoubleComplex, hipblasZscal);
    MAP2CF_V2(hipblasScal, hipblasComplex, float, hipblasCsscal);
    MAP2CF_V2(hipblasScal, hipblasDoubleComplex, double, hipblasZdscal);

    MAP2CF(hipblasScalBatched, float, float, hipblasSscalBatched);
    MAP2CF(hipblasScalBatched, double, double, hipblasDscalBatched);
    MAP2CF_V2(hipblasScalBatched, hipblasComplex, hipblasComplex, hipblasCscalBatched);
    MAP2CF_V2(hipblasScalBatched, hipblasDoubleComplex, hipblasDoubleComplex, hipblasZscalBatched);
    MAP2CF_V2(hipblasScalBatched, hipblasComplex, float, hipblasCsscalBatched);
    MAP2CF_V2(hipblasScalBatched, hipblasDoubleComplex, double, hipblasZdscalBatched);

    MAP2CF(hipblasScalStridedBatched, float, float, hipblasSscalStridedBatched);
    MAP2CF(hipblasScalStridedBatched, double, double, hipblasDscalStridedBatched);
    MAP2CF_V2(hipblasScalStridedBatched,
              hipblasComplex,
              hipblasComplex,
              hipblasCscalStridedBatched);
    MAP2CF_V2(hipblasScalStridedBatched,
              hipblasDoubleComplex,
              hipblasDoubleComplex,
              hipblasZscalStridedBatched);
    MAP2CF_V2(hipblasScalStridedBatched, hipblasComplex, float, hipblasCsscalStridedBatched);
    MAP2CF_V2(hipblasScalStridedBatched, hipblasDoubleComplex, double, hipblasZdscalStridedBatched);

    // Copy
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopy)(
        hipblasHandle_t handle, int n, const T* x, int incx, T* y, int incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyBatched)(hipblasHandle_t handle,
                                          int             n,
                                          const T* const  x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T*        x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count);

    MAP2CF(hipblasCopy, float, hipblasScopy);
    MAP2CF(hipblasCopy, double, hipblasDcopy);
    MAP2CF_V2(hipblasCopy, hipblasComplex, hipblasCcopy);
    MAP2CF_V2(hipblasCopy, hipblasDoubleComplex, hipblasZcopy);

    MAP2CF(hipblasCopyBatched, float, hipblasScopyBatched);
    MAP2CF(hipblasCopyBatched, double, hipblasDcopyBatched);
    MAP2CF_V2(hipblasCopyBatched, hipblasComplex, hipblasCcopyBatched);
    MAP2CF_V2(hipblasCopyBatched, hipblasDoubleComplex, hipblasZcopyBatched);

    MAP2CF(hipblasCopyStridedBatched, float, hipblasScopyStridedBatched);
    MAP2CF(hipblasCopyStridedBatched, double, hipblasDcopyStridedBatched);
    MAP2CF_V2(hipblasCopyStridedBatched, hipblasComplex, hipblasCcopyStridedBatched);
    MAP2CF_V2(hipblasCopyStridedBatched, hipblasDoubleComplex, hipblasZcopyStridedBatched);

    // Swap
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwap)(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapBatched)(hipblasHandle_t handle,
                                          int             n,
                                          T* const        x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 T*              x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count);

    MAP2CF(hipblasSwap, float, hipblasSswap);
    MAP2CF(hipblasSwap, double, hipblasDswap);
    MAP2CF_V2(hipblasSwap, hipblasComplex, hipblasCswap);
    MAP2CF_V2(hipblasSwap, hipblasDoubleComplex, hipblasZswap);

    MAP2CF(hipblasSwapBatched, float, hipblasSswapBatched);
    MAP2CF(hipblasSwapBatched, double, hipblasDswapBatched);
    MAP2CF_V2(hipblasSwapBatched, hipblasComplex, hipblasCswapBatched);
    MAP2CF_V2(hipblasSwapBatched, hipblasDoubleComplex, hipblasZswapBatched);

    MAP2CF(hipblasSwapStridedBatched, float, hipblasSswapStridedBatched);
    MAP2CF(hipblasSwapStridedBatched, double, hipblasDswapStridedBatched);
    MAP2CF_V2(hipblasSwapStridedBatched, hipblasComplex, hipblasCswapStridedBatched);
    MAP2CF_V2(hipblasSwapStridedBatched, hipblasDoubleComplex, hipblasZswapStridedBatched);

    // Dot
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDot)(
        hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotc)(
        hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotBatched)(hipblasHandle_t handle,
                                         int             n,
                                         const T* const  x[],
                                         int             incx,
                                         const T* const  y[],
                                         int             incy,
                                         int             batch_count,
                                         T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcBatched)(hipblasHandle_t handle,
                                          int             n,
                                          const T* const  x[],
                                          int             incx,
                                          const T* const  y[],
                                          int             incy,
                                          int             batch_count,
                                          T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotStridedBatched)(hipblasHandle_t handle,
                                                int             n,
                                                const T*        x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                const T*        y,
                                                int             incy,
                                                hipblasStride   stridey,
                                                int             batch_count,
                                                T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T*        x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 const T*        y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count,
                                                 T*              result);

    MAP2CF(hipblasDot, hipblasHalf, hipblasHdot);
    MAP2CF(hipblasDot, hipblasBfloat16, hipblasBfdot);
    MAP2CF(hipblasDot, float, hipblasSdot);
    MAP2CF(hipblasDot, double, hipblasDdot);
    MAP2CF_V2(hipblasDot, hipblasComplex, hipblasCdotu);
    MAP2CF_V2(hipblasDot, hipblasDoubleComplex, hipblasZdotu);
    MAP2CF_V2(hipblasDotc, hipblasComplex, hipblasCdotc);
    MAP2CF_V2(hipblasDotc, hipblasDoubleComplex, hipblasZdotc);

    MAP2CF(hipblasDotBatched, hipblasHalf, hipblasHdotBatched);
    MAP2CF(hipblasDotBatched, hipblasBfloat16, hipblasBfdotBatched);
    MAP2CF(hipblasDotBatched, float, hipblasSdotBatched);
    MAP2CF(hipblasDotBatched, double, hipblasDdotBatched);
    MAP2CF_V2(hipblasDotBatched, hipblasComplex, hipblasCdotuBatched);
    MAP2CF_V2(hipblasDotBatched, hipblasDoubleComplex, hipblasZdotuBatched);
    MAP2CF_V2(hipblasDotcBatched, hipblasComplex, hipblasCdotcBatched);
    MAP2CF_V2(hipblasDotcBatched, hipblasDoubleComplex, hipblasZdotcBatched);

    MAP2CF(hipblasDotStridedBatched, hipblasHalf, hipblasHdotStridedBatched);
    MAP2CF(hipblasDotStridedBatched, hipblasBfloat16, hipblasBfdotStridedBatched);
    MAP2CF(hipblasDotStridedBatched, float, hipblasSdotStridedBatched);
    MAP2CF(hipblasDotStridedBatched, double, hipblasDdotStridedBatched);
    MAP2CF_V2(hipblasDotStridedBatched, hipblasComplex, hipblasCdotuStridedBatched);
    MAP2CF_V2(hipblasDotStridedBatched, hipblasDoubleComplex, hipblasZdotuStridedBatched);
    MAP2CF_V2(hipblasDotcStridedBatched, hipblasComplex, hipblasCdotcStridedBatched);
    MAP2CF_V2(hipblasDotcStridedBatched, hipblasDoubleComplex, hipblasZdotcStridedBatched);

    // Asum
    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsum)(
        hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumBatched)(
        hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T1*       x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 int             batch_count,
                                                 T2*             result);

    MAP2CF(hipblasAsum, float, float, hipblasSasum);
    MAP2CF(hipblasAsum, double, double, hipblasDasum);
    MAP2CF_V2(hipblasAsum, hipblasComplex, float, hipblasScasum);
    MAP2CF_V2(hipblasAsum, hipblasDoubleComplex, double, hipblasDzasum);

    MAP2CF(hipblasAsumBatched, float, float, hipblasSasumBatched);
    MAP2CF(hipblasAsumBatched, double, double, hipblasDasumBatched);
    MAP2CF_V2(hipblasAsumBatched, hipblasComplex, float, hipblasScasumBatched);
    MAP2CF_V2(hipblasAsumBatched, hipblasDoubleComplex, double, hipblasDzasumBatched);

    MAP2CF(hipblasAsumStridedBatched, float, float, hipblasSasumStridedBatched);
    MAP2CF(hipblasAsumStridedBatched, double, double, hipblasDasumStridedBatched);
    MAP2CF_V2(hipblasAsumStridedBatched, hipblasComplex, float, hipblasScasumStridedBatched);
    MAP2CF_V2(hipblasAsumStridedBatched, hipblasDoubleComplex, double, hipblasDzasumStridedBatched);

    // nrm2
    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2)(
        hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2Batched)(
        hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2StridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T1*       x,
                                                 int             incxasum,
                                                 hipblasStride   stridex,
                                                 int             batch_count,
                                                 T2*             result);

    MAP2CF(hipblasNrm2, float, float, hipblasSnrm2);
    MAP2CF(hipblasNrm2, double, double, hipblasDnrm2);
    MAP2CF_V2(hipblasNrm2, hipblasComplex, float, hipblasScnrm2);
    MAP2CF_V2(hipblasNrm2, hipblasDoubleComplex, double, hipblasDznrm2);

    MAP2CF(hipblasNrm2Batched, float, float, hipblasSnrm2Batched);
    MAP2CF(hipblasNrm2Batched, double, double, hipblasDnrm2Batched);
    MAP2CF_V2(hipblasNrm2Batched, hipblasComplex, float, hipblasScnrm2Batched);
    MAP2CF_V2(hipblasNrm2Batched, hipblasDoubleComplex, double, hipblasDznrm2Batched);

    MAP2CF(hipblasNrm2StridedBatched, float, float, hipblasSnrm2StridedBatched);
    MAP2CF(hipblasNrm2StridedBatched, double, double, hipblasDnrm2StridedBatched);
    MAP2CF_V2(hipblasNrm2StridedBatched, hipblasComplex, float, hipblasScnrm2StridedBatched);
    MAP2CF_V2(hipblasNrm2StridedBatched, hipblasDoubleComplex, double, hipblasDznrm2StridedBatched);

    // Rot
    template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRot)(
        hipblasHandle_t handle, int n, T1* x, int incx, T1* y, int incy, const T2* c, const T3* s);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotBatched)(hipblasHandle_t handle,
                                         int             n,
                                         T1* const       x[],
                                         int             incx,
                                         T1* const       y[],
                                         int             incy,
                                         const T2*       c,
                                         const T3*       s,
                                         int             batch_count);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotStridedBatched)(hipblasHandle_t handle,
                                                int             n,
                                                T1*             x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                T1*             y,
                                                int             incy,
                                                hipblasStride   stridey,
                                                const T2*       c,
                                                const T3*       s,
                                                int             batch_count);

    MAP2CF(hipblasRot, float, float, float, hipblasSrot);
    MAP2CF(hipblasRot, double, double, double, hipblasDrot);
    MAP2CF_V2(hipblasRot, hipblasComplex, float, hipblasComplex, hipblasCrot);
    MAP2CF_V2(hipblasRot, hipblasDoubleComplex, double, hipblasDoubleComplex, hipblasZrot);
    MAP2CF_V2(hipblasRot, hipblasComplex, float, float, hipblasCsrot);
    MAP2CF_V2(hipblasRot, hipblasDoubleComplex, double, double, hipblasZdrot);

    MAP2CF(hipblasRotBatched, float, float, float, hipblasSrotBatched);
    MAP2CF(hipblasRotBatched, double, double, double, hipblasDrotBatched);
    MAP2CF_V2(hipblasRotBatched, hipblasComplex, float, hipblasComplex, hipblasCrotBatched);
    MAP2CF_V2(
        hipblasRotBatched, hipblasDoubleComplex, double, hipblasDoubleComplex, hipblasZrotBatched);
    MAP2CF_V2(hipblasRotBatched, hipblasComplex, float, float, hipblasCsrotBatched);
    MAP2CF_V2(hipblasRotBatched, hipblasDoubleComplex, double, double, hipblasZdrotBatched);

    MAP2CF(hipblasRotStridedBatched, float, float, float, hipblasSrotStridedBatched);
    MAP2CF(hipblasRotStridedBatched, double, double, double, hipblasDrotStridedBatched);
    MAP2CF_V2(
        hipblasRotStridedBatched, hipblasComplex, float, hipblasComplex, hipblasCrotStridedBatched);
    MAP2CF_V2(hipblasRotStridedBatched,
              hipblasDoubleComplex,
              double,
              hipblasDoubleComplex,
              hipblasZrotStridedBatched);
    MAP2CF_V2(hipblasRotStridedBatched, hipblasComplex, float, float, hipblasCsrotStridedBatched);
    MAP2CF_V2(
        hipblasRotStridedBatched, hipblasDoubleComplex, double, double, hipblasZdrotStridedBatched);

    // Rotg
    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotg)(hipblasHandle_t handle, T1* a, T1* b, T2* c, T1* s);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgBatched)(hipblasHandle_t handle,
                                          T1* const       a[],
                                          T1* const       b[],
                                          T2* const       c[],
                                          T1* const       s[],
                                          int             batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgStridedBatched)(hipblasHandle_t handle,
                                                 T1*             a,
                                                 hipblasStride   stridea,
                                                 T1*             b,
                                                 hipblasStride   strideb,
                                                 T2*             c,
                                                 hipblasStride   stridec,
                                                 T1*             s,
                                                 hipblasStride   strides,
                                                 int             batch_count);

    MAP2CF(hipblasRotg, float, float, hipblasSrotg);
    MAP2CF(hipblasRotg, double, double, hipblasDrotg);
    MAP2CF_V2(hipblasRotg, hipblasComplex, float, hipblasCrotg);
    MAP2CF_V2(hipblasRotg, hipblasDoubleComplex, double, hipblasZrotg);

    MAP2CF(hipblasRotgBatched, float, float, hipblasSrotgBatched);
    MAP2CF(hipblasRotgBatched, double, double, hipblasDrotgBatched);
    MAP2CF_V2(hipblasRotgBatched, hipblasComplex, float, hipblasCrotgBatched);
    MAP2CF_V2(hipblasRotgBatched, hipblasDoubleComplex, double, hipblasZrotgBatched);

    MAP2CF(hipblasRotgStridedBatched, float, float, hipblasSrotgStridedBatched);
    MAP2CF(hipblasRotgStridedBatched, double, double, hipblasDrotgStridedBatched);
    MAP2CF_V2(hipblasRotgStridedBatched, hipblasComplex, float, hipblasCrotgStridedBatched);
    MAP2CF_V2(hipblasRotgStridedBatched, hipblasDoubleComplex, double, hipblasZrotgStridedBatched);

    // rotm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotm)(
        hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy, const T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmBatched)(hipblasHandle_t handle,
                                          int             n,
                                          T* const        x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          const T* const  param[],
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 T*              x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 const T*        param,
                                                 hipblasStride   strideparam,
                                                 int             batch_count);

    MAP2CF(hipblasRotm, float, hipblasSrotm);
    MAP2CF(hipblasRotm, double, hipblasDrotm);

    MAP2CF(hipblasRotmBatched, float, hipblasSrotmBatched);
    MAP2CF(hipblasRotmBatched, double, hipblasDrotmBatched);

    MAP2CF(hipblasRotmStridedBatched, float, hipblasSrotmStridedBatched);
    MAP2CF(hipblasRotmStridedBatched, double, hipblasDrotmStridedBatched);

    // rotmg
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmg)(
        hipblasHandle_t handle, T* d1, T* d2, T* x1, const T* y1, T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgBatched)(hipblasHandle_t handle,
                                           T* const        d1[],
                                           T* const        d2[],
                                           T* const        x1[],
                                           const T* const  y1[],
                                           T* const        param[],
                                           int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgStridedBatched)(hipblasHandle_t handle,
                                                  T*              d1,
                                                  hipblasStride   stride_d1,
                                                  T*              d2,
                                                  hipblasStride   stride_d2,
                                                  T*              x1,
                                                  hipblasStride   stride_x1,
                                                  const T*        y1,
                                                  hipblasStride   stride_y1,
                                                  T*              param,
                                                  hipblasStride   strideparam,
                                                  int             batch_count);

    MAP2CF(hipblasRotmg, float, hipblasSrotmg);
    MAP2CF(hipblasRotmg, double, hipblasDrotmg);

    MAP2CF(hipblasRotmgBatched, float, hipblasSrotmgBatched);
    MAP2CF(hipblasRotmgBatched, double, hipblasDrotmgBatched);

    MAP2CF(hipblasRotmgStridedBatched, float, hipblasSrotmgStridedBatched);
    MAP2CF(hipblasRotmgStridedBatched, double, hipblasDrotmgStridedBatched);

    // amax
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamax)(
        hipblasHandle_t handle, int n, const T* x, int incx, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxBatched)(
        hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxStridedBatched)(hipblasHandle_t handle,
                                                  int             n,
                                                  const T*        x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  int*            result);

    MAP2CF(hipblasIamax, float, hipblasIsamax);
    MAP2CF(hipblasIamax, double, hipblasIdamax);
    MAP2CF_V2(hipblasIamax, hipblasComplex, hipblasIcamax);
    MAP2CF_V2(hipblasIamax, hipblasDoubleComplex, hipblasIzamax);

    MAP2CF(hipblasIamaxBatched, float, hipblasIsamaxBatched);
    MAP2CF(hipblasIamaxBatched, double, hipblasIdamaxBatched);
    MAP2CF_V2(hipblasIamaxBatched, hipblasComplex, hipblasIcamaxBatched);
    MAP2CF_V2(hipblasIamaxBatched, hipblasDoubleComplex, hipblasIzamaxBatched);

    MAP2CF(hipblasIamaxStridedBatched, float, hipblasIsamaxStridedBatched);
    MAP2CF(hipblasIamaxStridedBatched, double, hipblasIdamaxStridedBatched);
    MAP2CF_V2(hipblasIamaxStridedBatched, hipblasComplex, hipblasIcamaxStridedBatched);
    MAP2CF_V2(hipblasIamaxStridedBatched, hipblasDoubleComplex, hipblasIzamaxStridedBatched);

    // amin
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamin)(
        hipblasHandle_t handle, int n, const T* x, int incx, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminBatched)(
        hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminStridedBatched)(hipblasHandle_t handle,
                                                  int             n,
                                                  const T*        x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  int*            result);

    MAP2CF(hipblasIamin, float, hipblasIsamin);
    MAP2CF(hipblasIamin, double, hipblasIdamin);
    MAP2CF_V2(hipblasIamin, hipblasComplex, hipblasIcamin);
    MAP2CF_V2(hipblasIamin, hipblasDoubleComplex, hipblasIzamin);

    MAP2CF(hipblasIaminBatched, float, hipblasIsaminBatched);
    MAP2CF(hipblasIaminBatched, double, hipblasIdaminBatched);
    MAP2CF_V2(hipblasIaminBatched, hipblasComplex, hipblasIcaminBatched);
    MAP2CF_V2(hipblasIaminBatched, hipblasDoubleComplex, hipblasIzaminBatched);

    MAP2CF(hipblasIaminStridedBatched, float, hipblasIsaminStridedBatched);
    MAP2CF(hipblasIaminStridedBatched, double, hipblasIdaminStridedBatched);
    MAP2CF_V2(hipblasIaminStridedBatched, hipblasComplex, hipblasIcaminStridedBatched);
    MAP2CF_V2(hipblasIaminStridedBatched, hipblasDoubleComplex, hipblasIzaminStridedBatched);

    // axpy
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpy)(
        hipblasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyBatched)(hipblasHandle_t handle,
                                          int             n,
                                          const T*        alpha,
                                          const T* const  x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T*        alpha,
                                                 const T*        x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count);

    MAP2CF(hipblasAxpy, hipblasHalf, hipblasHaxpy);
    MAP2CF(hipblasAxpy, float, hipblasSaxpy);
    MAP2CF(hipblasAxpy, double, hipblasDaxpy);
    MAP2CF_V2(hipblasAxpy, hipblasComplex, hipblasCaxpy);
    MAP2CF_V2(hipblasAxpy, hipblasDoubleComplex, hipblasZaxpy);

    MAP2CF(hipblasAxpyBatched, hipblasHalf, hipblasHaxpyBatched);
    MAP2CF(hipblasAxpyBatched, float, hipblasSaxpyBatched);
    MAP2CF(hipblasAxpyBatched, double, hipblasDaxpyBatched);
    MAP2CF_V2(hipblasAxpyBatched, hipblasComplex, hipblasCaxpyBatched);
    MAP2CF_V2(hipblasAxpyBatched, hipblasDoubleComplex, hipblasZaxpyBatched);

    MAP2CF(hipblasAxpyStridedBatched, hipblasHalf, hipblasHaxpyStridedBatched);
    MAP2CF(hipblasAxpyStridedBatched, float, hipblasSaxpyStridedBatched);
    MAP2CF(hipblasAxpyStridedBatched, double, hipblasDaxpyStridedBatched);
    MAP2CF_V2(hipblasAxpyStridedBatched, hipblasComplex, hipblasCaxpyStridedBatched);
    MAP2CF_V2(hipblasAxpyStridedBatched, hipblasDoubleComplex, hipblasZaxpyStridedBatched);
}

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
                                         hipblasStride   stridex,
                                         const T*        y,
                                         int             incy,
                                         hipblasStride   stridey,
                                         T*              A,
                                         int             lda,
                                         hipblasStride   strideA,
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
                                          hipblasStride     strideA,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          hipblasStride     stridey,
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
                                          hipblasStride     stride_a,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stride_x,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          hipblasStride     stride_y,
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
                                         hipblasStride     stridex,
                                         T*                A,
                                         int               lda,
                                         hipblasStride     strideA,
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
                                          hipblasStride     stridex,
                                          const T*          y,
                                          int               incy,
                                          hipblasStride     stridey,
                                          T*                A,
                                          int               lda,
                                          hipblasStride     strideA,
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
                                          hipblasStride     strideAP,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          hipblasStride     stridey,
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
                                         hipblasStride     stridex,
                                         T*                AP,
                                         hipblasStride     strideAP,
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
                                          hipblasStride     stridex,
                                          const T*          y,
                                          int               incy,
                                          hipblasStride     stridey,
                                          T*                AP,
                                          hipblasStride     strideAP,
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
                                          hipblasStride     strideA,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          hipblasStride     stridey,
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
                                          hipblasStride     strideAP,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          hipblasStride     stridey,
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
                                         hipblasStride     stridex,
                                         T*                AP,
                                         hipblasStride     strideAP,
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
                                          hipblasStride     stridex,
                                          const T*          y,
                                          int               incy,
                                          hipblasStride     stridey,
                                          T*                AP,
                                          hipblasStride     strideAP,
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
                                   T* const          y[],
                                   int               incy,
                                   int               batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasSymvStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T*          A,
                                          int               lda,
                                          hipblasStride     strideA,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          const T*          beta,
                                          T*                y,
                                          int               incy,
                                          hipblasStride     stridey,
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
                                         hipblasStride     stridex,
                                         T*                A,
                                         int               lda,
                                         hipblasStride     strideA,
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
                                          hipblasStride     stridex,
                                          const T*          y,
                                          int               incy,
                                          hipblasStride     stridey,
                                          T*                A,
                                          int               lda,
                                          hipblasStride     strideA,
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
                                          hipblasStride      stride_a,
                                          T*                 x,
                                          int                incx,
                                          hipblasStride      stride_x,
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
                                          hipblasStride      strideA,
                                          T*                 x,
                                          int                incx,
                                          hipblasStride      stridex,
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
                                          hipblasStride      strideAP,
                                          T*                 x,
                                          int                incx,
                                          hipblasStride      stridex,
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
                                          hipblasStride      strideAP,
                                          T*                 x,
                                          int                incx,
                                          hipblasStride      stridex,
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
                                          hipblasStride      stride_a,
                                          T*                 x,
                                          int                incx,
                                          hipblasStride      stride_x,
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
                                          hipblasStride      strideA,
                                          T*                 x,
                                          int                incx,
                                          hipblasStride      stridex,
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
                                          hipblasStride      stride_a,
                                          const T*           x,
                                          int                incx,
                                          hipblasStride      stride_x,
                                          const T*           beta,
                                          T*                 y,
                                          int                incy,
                                          hipblasStride      stride_y,
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
                                          hipblasStride      strideA,
                                          const T*           x,
                                          int                incx,
                                          hipblasStride      stridex,
                                          const T*           beta,
                                          T*                 y,
                                          int                incy,
                                          hipblasStride      stridey,
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
                                          hipblasStride      strideA,
                                          const U*           beta,
                                          T*                 C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                           hipblasStride      strideA,
                                           const T*           B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const U*           beta,
                                           T*                 C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                           hipblasStride      strideA,
                                           const T*           B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const U*           beta,
                                           T*                 C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                          hipblasStride     strideA,
                                          const T*          B,
                                          int               ldb,
                                          hipblasStride     strideB,
                                          const T*          beta,
                                          T*                C,
                                          int               ldc,
                                          hipblasStride     strideC,
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
                                          hipblasStride      strideA,
                                          const T*           beta,
                                          T*                 C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                           hipblasStride      strideA,
                                           const T*           B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const T*           beta,
                                           T*                 C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                           hipblasStride      strideA,
                                           const T*           B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           const T*           beta,
                                           T*                 C,
                                           int                ldc,
                                           hipblasStride      strideC,
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
                                          hipblasStride      strideA,
                                          const T*           beta,
                                          const T*           B,
                                          int                ldb,
                                          hipblasStride      strideB,
                                          T*                 C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                                          hipblasStride     strideA,
                                          const T*          B,
                                          int               ldb,
                                          hipblasStride     strideB,
                                          const T*          beta,
                                          T*                C,
                                          int               ldc,
                                          hipblasStride     strideC,
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
                            const T*           B,
                            int                ldb,
                            T*                 C,
                            int                ldc);

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
                                   const T* const     B[],
                                   int                ldb,
                                   T* const           C[],
                                   int                ldc,
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
                                          hipblasStride      strideA,
                                          const T*           B,
                                          int                ldb,
                                          hipblasStride      strideB,
                                          T*                 C,
                                          int                ldc,
                                          hipblasStride      strideC,
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
                            const T*           A,
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
                                   const T* const     A[],
                                   int                lda,
                                   T* const           B[],
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
                                          const T*           A,
                                          int                lda,
                                          hipblasStride      strideA,
                                          T*                 B,
                                          int                ldb,
                                          hipblasStride      strideB,
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
hipblasStatus_t hipblasGetrfStridedBatched(hipblasHandle_t     handle,
                                           const int           n,
                                           T*                  A,
                                           const int           lda,
                                           const hipblasStride strideA,
                                           int*                ipiv,
                                           const hipblasStride strideP,
                                           int*                info,
                                           const int           batchCount);

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
                                           const hipblasStride      strideA,
                                           const int*               ipiv,
                                           const hipblasStride      strideP,
                                           T*                       B,
                                           const int                ldb,
                                           const hipblasStride      strideB,
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
hipblasStatus_t hipblasGeqrfStridedBatched(hipblasHandle_t     handle,
                                           const int           m,
                                           const int           n,
                                           T*                  A,
                                           const int           lda,
                                           const hipblasStride strideA,
                                           T*                  ipiv,
                                           const hipblasStride strideP,
                                           int*                info,
                                           const int           batchCount);

// gels
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGels(hipblasHandle_t    handle,
                            hipblasOperation_t trans,
                            const int          m,
                            const int          n,
                            const int          nrhs,
                            T*                 A,
                            const int          lda,
                            T*                 B,
                            const int          ldb,
                            int*               info,
                            int*               deviceInfo);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGelsBatched(hipblasHandle_t    handle,
                                   hipblasOperation_t trans,
                                   const int          m,
                                   const int          n,
                                   const int          nrhs,
                                   T* const           A[],
                                   const int          lda,
                                   T* const           B[],
                                   const int          ldb,
                                   int*               info,
                                   int*               deviceInfo,
                                   const int          batchCount);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGelsStridedBatched(hipblasHandle_t     handle,
                                          hipblasOperation_t  trans,
                                          const int           m,
                                          const int           n,
                                          const int           nrhs,
                                          T*                  A,
                                          const int           lda,
                                          const hipblasStride strideA,
                                          T*                  B,
                                          const int           ldb,
                                          const hipblasStride strideB,
                                          int*                info,
                                          int*                deviceInfo,
                                          const int           batchCount);

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
                                          hipblasStride     stride_A,
                                          const T*          x,
                                          int               incx,
                                          hipblasStride     stride_x,
                                          T*                C,
                                          int               ldc,
                                          hipblasStride     stride_C,
                                          int               batch_count);

// trtri
template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrtri(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             hipblasDiagType_t diag,
                             int               n,
                             const T*          A,
                             int               lda,
                             T*                invA,
                             int               ldinvA);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrtriBatched(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    hipblasDiagType_t diag,
                                    int               n,
                                    const T* const    A[],
                                    int               lda,
                                    T*                invA[],
                                    int               ldinvA,
                                    int               batch_count);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasTrtriStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           hipblasDiagType_t diag,
                                           int               n,
                                           const T*          A,
                                           int               lda,
                                           hipblasStride     stride_A,
                                           T*                invA,
                                           int               ldinvA,
                                           hipblasStride     stride_invA,
                                           int               batch_count);

#endif // _ROCBLAS_HPP_
