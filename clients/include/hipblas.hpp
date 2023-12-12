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
#ifdef HIPBLAS_V2
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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                 int                   lda);

hipblasStatus_t hipblasCgercCast(hipblasHandle_t       handle,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       A,
                                 int                   lda);

hipblasStatus_t hipblasZgeruCast(hipblasHandle_t             handle,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

hipblasStatus_t hipblasZgercCast(hipblasHandle_t             handle,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

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
                                        int                         batch_count);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                               int                         batch_count);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

// her
hipblasStatus_t hipblasCherCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda);

hipblasStatus_t hipblasZherCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda);

hipblasStatus_t hipblasCherBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batchCount);

hipblasStatus_t hipblasZherBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batchCount);

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
                                              int                   batchCount);

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
                                              int                         batchCount);

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
                                 int                   lda);

hipblasStatus_t hipblasZher2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                 int                   incy);

hipblasStatus_t hipblasZhpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* AP,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// hpr
hipblasStatus_t hipblasChprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP);

hipblasStatus_t hipblasZhprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasChprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount);

hipblasStatus_t hipblasZhprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount);

hipblasStatus_t hipblasChprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const float*          alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount);

hipblasStatus_t hipblasZhprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const double*               alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount);

// hpr2
hipblasStatus_t hipblasChpr2Cast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       AP);

hipblasStatus_t hipblasZhpr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasChpr2BatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       AP[],
                                        int                         batchCount);

hipblasStatus_t hipblasZhpr2BatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       AP[],
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// sbmv, spmv, spr2 no complex versions

// spr
hipblasStatus_t hipblasCsprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP);

hipblasStatus_t hipblasZsprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasCsprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount);

hipblasStatus_t hipblasZsprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount);

hipblasStatus_t hipblasCsprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const hipblasComplex* alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount);

hipblasStatus_t hipblasZsprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const hipblasDoubleComplex* alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// syr
hipblasStatus_t hipblasCsyrCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda);

hipblasStatus_t hipblasZsyrCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda);

hipblasStatus_t hipblasCsyrBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batch_count);

hipblasStatus_t hipblasZsyrBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batch_count);

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
                                              int                   batch_count);

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
                                              int                         batch_count);

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
                                 int                   lda);

hipblasStatus_t hipblasZsyr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// trsv
hipblasStatus_t hipblasCtrsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtrsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtrsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count);

hipblasStatus_t hipblasZtrsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                 int                   incx);

hipblasStatus_t hipblasZtbmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 int                         k,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                 int                   incx);

hipblasStatus_t hipblasZtbsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// tpmv
hipblasStatus_t hipblasCtpmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtpmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount);

hipblasStatus_t hipblasZtpmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// tpsv
hipblasStatus_t hipblasCtpsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtpsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtpsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount);

hipblasStatus_t hipblasZtpsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// trmv
hipblasStatus_t hipblasCtrmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtrmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtrmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count);

hipblasStatus_t hipblasZtrmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

#endif

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

    // ger
    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGer)(hipblasHandle_t handle,
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
    hipblasStatus_t (*hipblasGerBatched)(hipblasHandle_t handle,
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
    hipblasStatus_t (*hipblasGerStridedBatched)(hipblasHandle_t handle,
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

    MAP2CF(hipblasGer, float, false, hipblasSger);
    MAP2CF(hipblasGer, double, false, hipblasDger);
    MAP2CF_V2(hipblasGer, hipblasComplex, false, hipblasCgeru);
    MAP2CF_V2(hipblasGer, hipblasDoubleComplex, false, hipblasZgeru);
    MAP2CF_V2(hipblasGer, hipblasComplex, true, hipblasCgerc);
    MAP2CF_V2(hipblasGer, hipblasDoubleComplex, true, hipblasZgerc);

    MAP2CF(hipblasGerBatched, float, false, hipblasSgerBatched);
    MAP2CF(hipblasGerBatched, double, false, hipblasDgerBatched);
    MAP2CF_V2(hipblasGerBatched, hipblasComplex, false, hipblasCgeruBatched);
    MAP2CF_V2(hipblasGerBatched, hipblasDoubleComplex, false, hipblasZgeruBatched);
    MAP2CF_V2(hipblasGerBatched, hipblasComplex, true, hipblasCgercBatched);
    MAP2CF_V2(hipblasGerBatched, hipblasDoubleComplex, true, hipblasZgercBatched);

    MAP2CF(hipblasGerStridedBatched, float, false, hipblasSgerStridedBatched);
    MAP2CF(hipblasGerStridedBatched, double, false, hipblasDgerStridedBatched);
    MAP2CF_V2(hipblasGerStridedBatched, hipblasComplex, false, hipblasCgeruStridedBatched);
    MAP2CF_V2(hipblasGerStridedBatched, hipblasDoubleComplex, false, hipblasZgeruStridedBatched);
    MAP2CF_V2(hipblasGerStridedBatched, hipblasComplex, true, hipblasCgercStridedBatched);
    MAP2CF_V2(hipblasGerStridedBatched, hipblasDoubleComplex, true, hipblasZgercStridedBatched);

    // hbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmv)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHbmvBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHbmvStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF_V2(hipblasHbmv, hipblasComplex, hipblasChbmv);
    MAP2CF_V2(hipblasHbmv, hipblasDoubleComplex, hipblasZhbmv);

    MAP2CF_V2(hipblasHbmvBatched, hipblasComplex, hipblasChbmvBatched);
    MAP2CF_V2(hipblasHbmvBatched, hipblasDoubleComplex, hipblasZhbmvBatched);

    MAP2CF_V2(hipblasHbmvStridedBatched, hipblasComplex, hipblasChbmvStridedBatched);
    MAP2CF_V2(hipblasHbmvStridedBatched, hipblasDoubleComplex, hipblasZhbmvStridedBatched);

    // hemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemv)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHemvBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHemvStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF_V2(hipblasHemv, hipblasComplex, hipblasChemv);
    MAP2CF_V2(hipblasHemv, hipblasDoubleComplex, hipblasZhemv);

    MAP2CF_V2(hipblasHemvBatched, hipblasComplex, hipblasChemvBatched);
    MAP2CF_V2(hipblasHemvBatched, hipblasDoubleComplex, hipblasZhemvBatched);

    MAP2CF_V2(hipblasHemvStridedBatched, hipblasComplex, hipblasChemvStridedBatched);
    MAP2CF_V2(hipblasHemvStridedBatched, hipblasDoubleComplex, hipblasZhemvStridedBatched);

    // her
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const U*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                A,
                                  int               lda);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const U*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          A[],
                                         int               lda,
                                         int               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF_V2(hipblasHer, hipblasComplex, float, hipblasCher);
    MAP2CF_V2(hipblasHer, hipblasDoubleComplex, double, hipblasZher);

    MAP2CF_V2(hipblasHerBatched, hipblasComplex, float, hipblasCherBatched);
    MAP2CF_V2(hipblasHerBatched, hipblasDoubleComplex, double, hipblasZherBatched);

    MAP2CF_V2(hipblasHerStridedBatched, hipblasComplex, float, hipblasCherStridedBatched);
    MAP2CF_V2(hipblasHerStridedBatched, hipblasDoubleComplex, double, hipblasZherStridedBatched);

    // her2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHer2Batched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHer2StridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF_V2(hipblasHer2, hipblasComplex, hipblasCher2);
    MAP2CF_V2(hipblasHer2, hipblasDoubleComplex, hipblasZher2);

    MAP2CF_V2(hipblasHer2Batched, hipblasComplex, hipblasCher2Batched);
    MAP2CF_V2(hipblasHer2Batched, hipblasDoubleComplex, hipblasZher2Batched);

    MAP2CF_V2(hipblasHer2StridedBatched, hipblasComplex, hipblasCher2StridedBatched);
    MAP2CF_V2(hipblasHer2StridedBatched, hipblasDoubleComplex, hipblasZher2StridedBatched);

    // hpmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmv)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHpmvBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHpmvStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF_V2(hipblasHpmv, hipblasComplex, hipblasChpmv);
    MAP2CF_V2(hipblasHpmv, hipblasDoubleComplex, hipblasZhpmv);

    MAP2CF_V2(hipblasHpmvBatched, hipblasComplex, hipblasChpmvBatched);
    MAP2CF_V2(hipblasHpmvBatched, hipblasDoubleComplex, hipblasZhpmvBatched);

    MAP2CF_V2(hipblasHpmvStridedBatched, hipblasComplex, hipblasChpmvStridedBatched);
    MAP2CF_V2(hipblasHpmvStridedBatched, hipblasDoubleComplex, hipblasZhpmvStridedBatched);

    // hpr
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const U*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                AP);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const U*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          AP[],
                                         int               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprStridedBatched)(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const U*          alpha,
                                                const T*          x,
                                                int               incx,
                                                hipblasStride     stridex,
                                                T*                AP,
                                                hipblasStride     strideAP,
                                                int               batchCount);

    MAP2CF_V2(hipblasHpr, hipblasComplex, float, hipblasChpr);
    MAP2CF_V2(hipblasHpr, hipblasDoubleComplex, double, hipblasZhpr);

    MAP2CF_V2(hipblasHprBatched, hipblasComplex, float, hipblasChprBatched);
    MAP2CF_V2(hipblasHprBatched, hipblasDoubleComplex, double, hipblasZhprBatched);

    MAP2CF_V2(hipblasHprStridedBatched, hipblasComplex, float, hipblasChprStridedBatched);
    MAP2CF_V2(hipblasHprStridedBatched, hipblasDoubleComplex, double, hipblasZhprStridedBatched);

    // hpr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          x,
                                   int               incx,
                                   const T*          y,
                                   int               incy,
                                   T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2Batched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHpr2StridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF_V2(hipblasHpr2, hipblasComplex, hipblasChpr2);
    MAP2CF_V2(hipblasHpr2, hipblasDoubleComplex, hipblasZhpr2);

    MAP2CF_V2(hipblasHpr2Batched, hipblasComplex, hipblasChpr2Batched);
    MAP2CF_V2(hipblasHpr2Batched, hipblasDoubleComplex, hipblasZhpr2Batched);

    MAP2CF_V2(hipblasHpr2StridedBatched, hipblasComplex, hipblasChpr2StridedBatched);
    MAP2CF_V2(hipblasHpr2StridedBatched, hipblasDoubleComplex, hipblasZhpr2StridedBatched);

    // sbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmv)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSbmvBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSbmvStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasSbmv, float, hipblasSsbmv);
    MAP2CF(hipblasSbmv, double, hipblasDsbmv);

    MAP2CF(hipblasSbmvBatched, float, hipblasSsbmvBatched);
    MAP2CF(hipblasSbmvBatched, double, hipblasDsbmvBatched);

    MAP2CF(hipblasSbmvStridedBatched, float, hipblasSsbmvStridedBatched);
    MAP2CF(hipblasSbmvStridedBatched, double, hipblasDsbmvStridedBatched);

    // spmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmv)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSpmvBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSpmvStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasSpmv, float, hipblasSspmv);
    MAP2CF(hipblasSpmv, double, hipblasDspmv);

    MAP2CF(hipblasSpmvBatched, float, hipblasSspmvBatched);
    MAP2CF(hipblasSpmvBatched, double, hipblasDspmvBatched);

    MAP2CF(hipblasSpmvStridedBatched, float, hipblasSspmvStridedBatched);
    MAP2CF(hipblasSpmvStridedBatched, double, hipblasDspmvStridedBatched);

    // spr
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const T*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          AP[],
                                         int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprStridedBatched)(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const T*          alpha,
                                                const T*          x,
                                                int               incx,
                                                hipblasStride     stridex,
                                                T*                AP,
                                                hipblasStride     strideAP,
                                                int               batchCount);

    MAP2CF(hipblasSpr, float, hipblasSspr);
    MAP2CF(hipblasSpr, double, hipblasDspr);
    MAP2CF_V2(hipblasSpr, hipblasComplex, hipblasCspr);
    MAP2CF_V2(hipblasSpr, hipblasDoubleComplex, hipblasZspr);

    MAP2CF(hipblasSprBatched, float, hipblasSsprBatched);
    MAP2CF(hipblasSprBatched, double, hipblasDsprBatched);
    MAP2CF_V2(hipblasSprBatched, hipblasComplex, hipblasCsprBatched);
    MAP2CF_V2(hipblasSprBatched, hipblasDoubleComplex, hipblasZsprBatched);

    MAP2CF(hipblasSprStridedBatched, float, hipblasSsprStridedBatched);
    MAP2CF(hipblasSprStridedBatched, double, hipblasDsprStridedBatched);
    MAP2CF_V2(hipblasSprStridedBatched, hipblasComplex, hipblasCsprStridedBatched);
    MAP2CF_V2(hipblasSprStridedBatched, hipblasDoubleComplex, hipblasZsprStridedBatched);

    // spr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          x,
                                   int               incx,
                                   const T*          y,
                                   int               incy,
                                   T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2Batched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSpr2StridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasSpr2, float, hipblasSspr2);
    MAP2CF(hipblasSpr2, double, hipblasDspr2);

    MAP2CF(hipblasSpr2Batched, float, hipblasSspr2Batched);
    MAP2CF(hipblasSpr2Batched, double, hipblasDspr2Batched);

    MAP2CF(hipblasSpr2StridedBatched, float, hipblasSspr2StridedBatched);
    MAP2CF(hipblasSpr2StridedBatched, double, hipblasDspr2StridedBatched);

    // symv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymv)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSymvBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSymvStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasSymv, float, hipblasSsymv);
    MAP2CF(hipblasSymv, double, hipblasDsymv);
    MAP2CF_V2(hipblasSymv, hipblasComplex, hipblasCsymv);
    MAP2CF_V2(hipblasSymv, hipblasDoubleComplex, hipblasZsymv);

    MAP2CF(hipblasSymvBatched, float, hipblasSsymvBatched);
    MAP2CF(hipblasSymvBatched, double, hipblasDsymvBatched);
    MAP2CF_V2(hipblasSymvBatched, hipblasComplex, hipblasCsymvBatched);
    MAP2CF_V2(hipblasSymvBatched, hipblasDoubleComplex, hipblasZsymvBatched);

    MAP2CF(hipblasSymvStridedBatched, float, hipblasSsymvStridedBatched);
    MAP2CF(hipblasSymvStridedBatched, double, hipblasDsymvStridedBatched);
    MAP2CF_V2(hipblasSymvStridedBatched, hipblasComplex, hipblasCsymvStridedBatched);
    MAP2CF_V2(hipblasSymvStridedBatched, hipblasDoubleComplex, hipblasZsymvStridedBatched);

    // syr
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                A,
                                  int               lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const T*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          A[],
                                         int               lda,
                                         int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasSyr, float, hipblasSsyr);
    MAP2CF(hipblasSyr, double, hipblasDsyr);
    MAP2CF_V2(hipblasSyr, hipblasComplex, hipblasCsyr);
    MAP2CF_V2(hipblasSyr, hipblasDoubleComplex, hipblasZsyr);

    MAP2CF(hipblasSyrBatched, float, hipblasSsyrBatched);
    MAP2CF(hipblasSyrBatched, double, hipblasDsyrBatched);
    MAP2CF_V2(hipblasSyrBatched, hipblasComplex, hipblasCsyrBatched);
    MAP2CF_V2(hipblasSyrBatched, hipblasDoubleComplex, hipblasZsyrBatched);

    MAP2CF(hipblasSyrStridedBatched, float, hipblasSsyrStridedBatched);
    MAP2CF(hipblasSyrStridedBatched, double, hipblasDsyrStridedBatched);
    MAP2CF_V2(hipblasSyrStridedBatched, hipblasComplex, hipblasCsyrStridedBatched);
    MAP2CF_V2(hipblasSyrStridedBatched, hipblasDoubleComplex, hipblasZsyrStridedBatched);

    // syr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSyr2Batched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSyr2StridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasSyr2, float, hipblasSsyr2);
    MAP2CF(hipblasSyr2, double, hipblasDsyr2);
    MAP2CF_V2(hipblasSyr2, hipblasComplex, hipblasCsyr2);
    MAP2CF_V2(hipblasSyr2, hipblasDoubleComplex, hipblasZsyr2);

    MAP2CF(hipblasSyr2Batched, float, hipblasSsyr2Batched);
    MAP2CF(hipblasSyr2Batched, double, hipblasDsyr2Batched);
    MAP2CF_V2(hipblasSyr2Batched, hipblasComplex, hipblasCsyr2Batched);
    MAP2CF_V2(hipblasSyr2Batched, hipblasDoubleComplex, hipblasZsyr2Batched);

    MAP2CF(hipblasSyr2StridedBatched, float, hipblasSsyr2StridedBatched);
    MAP2CF(hipblasSyr2StridedBatched, double, hipblasDsyr2StridedBatched);
    MAP2CF_V2(hipblasSyr2StridedBatched, hipblasComplex, hipblasCsyr2StridedBatched);
    MAP2CF_V2(hipblasSyr2StridedBatched, hipblasDoubleComplex, hipblasZsyr2StridedBatched);

    // tbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmv)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTbmvBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTbmvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasTbmv, float, hipblasStbmv);
    MAP2CF(hipblasTbmv, double, hipblasDtbmv);
    MAP2CF_V2(hipblasTbmv, hipblasComplex, hipblasCtbmv);
    MAP2CF_V2(hipblasTbmv, hipblasDoubleComplex, hipblasZtbmv);

    MAP2CF(hipblasTbmvBatched, float, hipblasStbmvBatched);
    MAP2CF(hipblasTbmvBatched, double, hipblasDtbmvBatched);
    MAP2CF_V2(hipblasTbmvBatched, hipblasComplex, hipblasCtbmvBatched);
    MAP2CF_V2(hipblasTbmvBatched, hipblasDoubleComplex, hipblasZtbmvBatched);

    MAP2CF(hipblasTbmvStridedBatched, float, hipblasStbmvStridedBatched);
    MAP2CF(hipblasTbmvStridedBatched, double, hipblasDtbmvStridedBatched);
    MAP2CF_V2(hipblasTbmvStridedBatched, hipblasComplex, hipblasCtbmvStridedBatched);
    MAP2CF_V2(hipblasTbmvStridedBatched, hipblasDoubleComplex, hipblasZtbmvStridedBatched);

    // tbsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsv)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTbsvBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTbsvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasTbsv, float, hipblasStbsv);
    MAP2CF(hipblasTbsv, double, hipblasDtbsv);
    MAP2CF_V2(hipblasTbsv, hipblasComplex, hipblasCtbsv);
    MAP2CF_V2(hipblasTbsv, hipblasDoubleComplex, hipblasZtbsv);

    MAP2CF(hipblasTbsvBatched, float, hipblasStbsvBatched);
    MAP2CF(hipblasTbsvBatched, double, hipblasDtbsvBatched);
    MAP2CF_V2(hipblasTbsvBatched, hipblasComplex, hipblasCtbsvBatched);
    MAP2CF_V2(hipblasTbsvBatched, hipblasDoubleComplex, hipblasZtbsvBatched);

    MAP2CF(hipblasTbsvStridedBatched, float, hipblasStbsvStridedBatched);
    MAP2CF(hipblasTbsvStridedBatched, double, hipblasDtbsvStridedBatched);
    MAP2CF_V2(hipblasTbsvStridedBatched, hipblasComplex, hipblasCtbsvStridedBatched);
    MAP2CF_V2(hipblasTbsvStridedBatched, hipblasDoubleComplex, hipblasZtbsvStridedBatched);

    // tpmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           AP,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T* const     AP[],
                                          T* const           x[],
                                          int                incx,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasTpmv, float, hipblasStpmv);
    MAP2CF(hipblasTpmv, double, hipblasDtpmv);
    MAP2CF_V2(hipblasTpmv, hipblasComplex, hipblasCtpmv);
    MAP2CF_V2(hipblasTpmv, hipblasDoubleComplex, hipblasZtpmv);

    MAP2CF(hipblasTpmvBatched, float, hipblasStpmvBatched);
    MAP2CF(hipblasTpmvBatched, double, hipblasDtpmvBatched);
    MAP2CF_V2(hipblasTpmvBatched, hipblasComplex, hipblasCtpmvBatched);
    MAP2CF_V2(hipblasTpmvBatched, hipblasDoubleComplex, hipblasZtpmvBatched);

    MAP2CF(hipblasTpmvStridedBatched, float, hipblasStpmvStridedBatched);
    MAP2CF(hipblasTpmvStridedBatched, double, hipblasDtpmvStridedBatched);
    MAP2CF_V2(hipblasTpmvStridedBatched, hipblasComplex, hipblasCtpmvStridedBatched);
    MAP2CF_V2(hipblasTpmvStridedBatched, hipblasDoubleComplex, hipblasZtpmvStridedBatched);

    // tpsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           AP,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T* const     AP[],
                                          T* const           x[],
                                          int                incx,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasTpsv, float, hipblasStpsv);
    MAP2CF(hipblasTpsv, double, hipblasDtpsv);
    MAP2CF_V2(hipblasTpsv, hipblasComplex, hipblasCtpsv);
    MAP2CF_V2(hipblasTpsv, hipblasDoubleComplex, hipblasZtpsv);

    MAP2CF(hipblasTpsvBatched, float, hipblasStpsvBatched);
    MAP2CF(hipblasTpsvBatched, double, hipblasDtpsvBatched);
    MAP2CF_V2(hipblasTpsvBatched, hipblasComplex, hipblasCtpsvBatched);
    MAP2CF_V2(hipblasTpsvBatched, hipblasDoubleComplex, hipblasZtpsvBatched);

    MAP2CF(hipblasTpsvStridedBatched, float, hipblasStpsvStridedBatched);
    MAP2CF(hipblasTpsvStridedBatched, double, hipblasDtpsvStridedBatched);
    MAP2CF_V2(hipblasTpsvStridedBatched, hipblasComplex, hipblasCtpsvStridedBatched);
    MAP2CF_V2(hipblasTpsvStridedBatched, hipblasDoubleComplex, hipblasZtpsvStridedBatched);

    // trmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           A,
                                   int                lda,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTrmvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasTrmv, float, hipblasStrmv);
    MAP2CF(hipblasTrmv, double, hipblasDtrmv);
    MAP2CF_V2(hipblasTrmv, hipblasComplex, hipblasCtrmv);
    MAP2CF_V2(hipblasTrmv, hipblasDoubleComplex, hipblasZtrmv);

    MAP2CF(hipblasTrmvBatched, float, hipblasStrmvBatched);
    MAP2CF(hipblasTrmvBatched, double, hipblasDtrmvBatched);
    MAP2CF_V2(hipblasTrmvBatched, hipblasComplex, hipblasCtrmvBatched);
    MAP2CF_V2(hipblasTrmvBatched, hipblasDoubleComplex, hipblasZtrmvBatched);

    MAP2CF(hipblasTrmvStridedBatched, float, hipblasStrmvStridedBatched);
    MAP2CF(hipblasTrmvStridedBatched, double, hipblasDtrmvStridedBatched);
    MAP2CF_V2(hipblasTrmvStridedBatched, hipblasComplex, hipblasCtrmvStridedBatched);
    MAP2CF_V2(hipblasTrmvStridedBatched, hipblasDoubleComplex, hipblasZtrmvStridedBatched);

    // trsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           A,
                                   int                lda,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTrsvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasTrsv, float, hipblasStrsv);
    MAP2CF(hipblasTrsv, double, hipblasDtrsv);
    MAP2CF_V2(hipblasTrsv, hipblasComplex, hipblasCtrsv);
    MAP2CF_V2(hipblasTrsv, hipblasDoubleComplex, hipblasZtrsv);

    MAP2CF(hipblasTrsvBatched, float, hipblasStrsvBatched);
    MAP2CF(hipblasTrsvBatched, double, hipblasDtrsvBatched);
    MAP2CF_V2(hipblasTrsvBatched, hipblasComplex, hipblasCtrsvBatched);
    MAP2CF_V2(hipblasTrsvBatched, hipblasDoubleComplex, hipblasZtrsvBatched);

    MAP2CF(hipblasTrsvStridedBatched, float, hipblasStrsvStridedBatched);
    MAP2CF(hipblasTrsvStridedBatched, double, hipblasDtrsvStridedBatched);
    MAP2CF_V2(hipblasTrsvStridedBatched, hipblasComplex, hipblasCtrsvStridedBatched);
    MAP2CF_V2(hipblasTrsvStridedBatched, hipblasDoubleComplex, hipblasZtrsvStridedBatched);

    // gbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmv)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGbmvBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGbmvStridedBatched)(hipblasHandle_t    handle,
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

    MAP2CF(hipblasGbmv, float, hipblasSgbmv);
    MAP2CF(hipblasGbmv, double, hipblasDgbmv);
    MAP2CF_V2(hipblasGbmv, hipblasComplex, hipblasCgbmv);
    MAP2CF_V2(hipblasGbmv, hipblasDoubleComplex, hipblasZgbmv);

    MAP2CF(hipblasGbmvBatched, float, hipblasSgbmvBatched);
    MAP2CF(hipblasGbmvBatched, double, hipblasDgbmvBatched);
    MAP2CF_V2(hipblasGbmvBatched, hipblasComplex, hipblasCgbmvBatched);
    MAP2CF_V2(hipblasGbmvBatched, hipblasDoubleComplex, hipblasZgbmvBatched);

    MAP2CF(hipblasGbmvStridedBatched, float, hipblasSgbmvStridedBatched);
    MAP2CF(hipblasGbmvStridedBatched, double, hipblasDgbmvStridedBatched);
    MAP2CF_V2(hipblasGbmvStridedBatched, hipblasComplex, hipblasCgbmvStridedBatched);
    MAP2CF_V2(hipblasGbmvStridedBatched, hipblasDoubleComplex, hipblasZgbmvStridedBatched);

    // gemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemv)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGemvBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGemvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 int                m,
                                                 int                n,
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

    MAP2CF(hipblasGemv, float, hipblasSgemv);
    MAP2CF(hipblasGemv, double, hipblasDgemv);
    MAP2CF_V2(hipblasGemv, hipblasComplex, hipblasCgemv);
    MAP2CF_V2(hipblasGemv, hipblasDoubleComplex, hipblasZgemv);

    MAP2CF(hipblasGemvBatched, float, hipblasSgemvBatched);
    MAP2CF(hipblasGemvBatched, double, hipblasDgemvBatched);
    MAP2CF_V2(hipblasGemvBatched, hipblasComplex, hipblasCgemvBatched);
    MAP2CF_V2(hipblasGemvBatched, hipblasDoubleComplex, hipblasZgemvBatched);

    MAP2CF(hipblasGemvStridedBatched, float, hipblasSgemvStridedBatched);
    MAP2CF(hipblasGemvStridedBatched, double, hipblasDgemvStridedBatched);
    MAP2CF_V2(hipblasGemvStridedBatched, hipblasComplex, hipblasCgemvStridedBatched);
    MAP2CF_V2(hipblasGemvStridedBatched, hipblasDoubleComplex, hipblasZgemvStridedBatched);
}

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
