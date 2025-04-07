/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if !defined(WIN32) // WIN doesn't have fortran tests
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

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScal_64)(
        hipblasHandle_t handle, int64_t n, const U* alpha, T* x, int64_t incx);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const U*        alpha,
                                             T* const        x[],
                                             int64_t         incx,
                                             int64_t         batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const U*        alpha,
                                                    T*              x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasScal, float, float, hipblasSscal);
    MAP2CF_D64(hipblasScal, double, double, hipblasDscal);
    MAP2CF_D64(hipblasScal, hipComplex, hipComplex, hipblasCscal);
    MAP2CF_D64(hipblasScal, hipDoubleComplex, hipDoubleComplex, hipblasZscal);
    MAP2CF_D64(hipblasScal, hipComplex, float, hipblasCsscal);
    MAP2CF_D64(hipblasScal, hipDoubleComplex, double, hipblasZdscal);

    MAP2CF_D64(hipblasScalBatched, float, float, hipblasSscalBatched);
    MAP2CF_D64(hipblasScalBatched, double, double, hipblasDscalBatched);
    MAP2CF_D64(hipblasScalBatched, hipComplex, hipComplex, hipblasCscalBatched);
    MAP2CF_D64(hipblasScalBatched,
                  hipDoubleComplex,
                  hipDoubleComplex,
                  hipblasZscalBatched);
    MAP2CF_D64(hipblasScalBatched, hipComplex, float, hipblasCsscalBatched);
    MAP2CF_D64(hipblasScalBatched, hipDoubleComplex, double, hipblasZdscalBatched);

    MAP2CF_D64(hipblasScalStridedBatched, float, float, hipblasSscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched, double, double, hipblasDscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched,
                  hipComplex,
                  hipComplex,
                  hipblasCscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched,
                  hipDoubleComplex,
                  hipDoubleComplex,
                  hipblasZscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched, hipComplex, float, hipblasCsscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZdscalStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopy_64)(
        hipblasHandle_t handle, int64_t n, const T* x, int64_t incx, T* y, int64_t incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T* const  x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T*        x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasCopy, float, hipblasScopy);
    MAP2CF_D64(hipblasCopy, double, hipblasDcopy);
    MAP2CF_D64(hipblasCopy, hipComplex, hipblasCcopy);
    MAP2CF_D64(hipblasCopy, hipDoubleComplex, hipblasZcopy);

    MAP2CF_D64(hipblasCopyBatched, float, hipblasScopyBatched);
    MAP2CF_D64(hipblasCopyBatched, double, hipblasDcopyBatched);
    MAP2CF_D64(hipblasCopyBatched, hipComplex, hipblasCcopyBatched);
    MAP2CF_D64(hipblasCopyBatched, hipDoubleComplex, hipblasZcopyBatched);

    MAP2CF_D64(hipblasCopyStridedBatched, float, hipblasScopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, double, hipblasDcopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, hipComplex, hipblasCcopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, hipDoubleComplex, hipblasZcopyStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwap_64)(
        hipblasHandle_t handle, int64_t n, T* x, int64_t incx, T* y, int64_t incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             T* const        x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    T*              x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasSwap, float, hipblasSswap);
    MAP2CF_D64(hipblasSwap, double, hipblasDswap);
    MAP2CF_D64(hipblasSwap, hipComplex, hipblasCswap);
    MAP2CF_D64(hipblasSwap, hipDoubleComplex, hipblasZswap);

    MAP2CF_D64(hipblasSwapBatched, float, hipblasSswapBatched);
    MAP2CF_D64(hipblasSwapBatched, double, hipblasDswapBatched);
    MAP2CF_D64(hipblasSwapBatched, hipComplex, hipblasCswapBatched);
    MAP2CF_D64(hipblasSwapBatched, hipDoubleComplex, hipblasZswapBatched);

    MAP2CF_D64(hipblasSwapStridedBatched, float, hipblasSswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, double, hipblasDswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, hipComplex, hipblasCswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, hipDoubleComplex, hipblasZswapStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDot_64)(hipblasHandle_t handle,
                                     int64_t         n,
                                     const T*        x,
                                     int64_t         incx,
                                     const T*        y,
                                     int64_t         incy,
                                     T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotc_64)(hipblasHandle_t handle,
                                      int64_t         n,
                                      const T*        x,
                                      int64_t         incx,
                                      const T*        y,
                                      int64_t         incy,
                                      T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotBatched_64)(hipblasHandle_t handle,
                                            int64_t         n,
                                            const T* const  x[],
                                            int64_t         incx,
                                            const T* const  y[],
                                            int64_t         incy,
                                            int64_t         batch_count,
                                            T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T* const  x[],
                                             int64_t         incx,
                                             const T* const  y[],
                                             int64_t         incy,
                                             int64_t         batch_count,
                                             T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotStridedBatched_64)(hipblasHandle_t handle,
                                                   int64_t         n,
                                                   const T*        x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   const T*        y,
                                                   int64_t         incy,
                                                   hipblasStride   stridey,
                                                   int64_t         batch_count,
                                                   T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T*        x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    const T*        y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count,
                                                    T*              result);

    MAP2CF_D64(hipblasDot, hipblasHalf, hipblasHdot);
    MAP2CF_D64(hipblasDot, hipblasBfloat16, hipblasBfdot);
    MAP2CF_D64(hipblasDot, float, hipblasSdot);
    MAP2CF_D64(hipblasDot, double, hipblasDdot);
    MAP2CF_D64(hipblasDot, hipComplex, hipblasCdotu);
    MAP2CF_D64(hipblasDot, hipDoubleComplex, hipblasZdotu);
    MAP2CF_D64(hipblasDotc, hipComplex, hipblasCdotc);
    MAP2CF_D64(hipblasDotc, hipDoubleComplex, hipblasZdotc);

    MAP2CF_D64(hipblasDotBatched, hipblasHalf, hipblasHdotBatched);
    MAP2CF_D64(hipblasDotBatched, hipblasBfloat16, hipblasBfdotBatched);
    MAP2CF_D64(hipblasDotBatched, float, hipblasSdotBatched);
    MAP2CF_D64(hipblasDotBatched, double, hipblasDdotBatched);
    MAP2CF_D64(hipblasDotBatched, hipComplex, hipblasCdotuBatched);
    MAP2CF_D64(hipblasDotBatched, hipDoubleComplex, hipblasZdotuBatched);
    MAP2CF_D64(hipblasDotcBatched, hipComplex, hipblasCdotcBatched);
    MAP2CF_D64(hipblasDotcBatched, hipDoubleComplex, hipblasZdotcBatched);

    MAP2CF_D64(hipblasDotStridedBatched, hipblasHalf, hipblasHdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, hipblasBfloat16, hipblasBfdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, float, hipblasSdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, double, hipblasDdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, hipComplex, hipblasCdotuStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, hipDoubleComplex, hipblasZdotuStridedBatched);
    MAP2CF_D64(hipblasDotcStridedBatched, hipComplex, hipblasCdotcStridedBatched);
    MAP2CF_D64(hipblasDotcStridedBatched, hipDoubleComplex, hipblasZdotcStridedBatched);

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

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsum_64)(
        hipblasHandle_t handle, int64_t n, const T1* x, int64_t incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T1* const x[],
                                             int64_t         incx,
                                             int64_t         batch_count,
                                             T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T1*       x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    int64_t         batch_count,
                                                    T2*             result);

    MAP2CF_D64(hipblasAsum, float, float, hipblasSasum);
    MAP2CF_D64(hipblasAsum, double, double, hipblasDasum);
    MAP2CF_D64(hipblasAsum, hipComplex, float, hipblasScasum);
    MAP2CF_D64(hipblasAsum, hipDoubleComplex, double, hipblasDzasum);

    MAP2CF_D64(hipblasAsumBatched, float, float, hipblasSasumBatched);
    MAP2CF_D64(hipblasAsumBatched, double, double, hipblasDasumBatched);
    MAP2CF_D64(hipblasAsumBatched, hipComplex, float, hipblasScasumBatched);
    MAP2CF_D64(hipblasAsumBatched, hipDoubleComplex, double, hipblasDzasumBatched);

    MAP2CF_D64(hipblasAsumStridedBatched, float, float, hipblasSasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched, double, double, hipblasDasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched, hipComplex, float, hipblasScasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasDzasumStridedBatched);

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
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 int             batch_count,
                                                 T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2_64)(
        hipblasHandle_t handle, int64_t n, const T1* x, int64_t incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2Batched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T1* const x[],
                                             int64_t         incx,
                                             int64_t         batch_count,
                                             T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2StridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T1*       x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    int64_t         batch_count,
                                                    T2*             result);

    MAP2CF_D64(hipblasNrm2, float, float, hipblasSnrm2);
    MAP2CF_D64(hipblasNrm2, double, double, hipblasDnrm2);
    MAP2CF_D64(hipblasNrm2, hipComplex, float, hipblasScnrm2);
    MAP2CF_D64(hipblasNrm2, hipDoubleComplex, double, hipblasDznrm2);

    MAP2CF_D64(hipblasNrm2Batched, float, float, hipblasSnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, double, double, hipblasDnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, hipComplex, float, hipblasScnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, hipDoubleComplex, double, hipblasDznrm2Batched);

    MAP2CF_D64(hipblasNrm2StridedBatched, float, float, hipblasSnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched, double, double, hipblasDnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched, hipComplex, float, hipblasScnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasDznrm2StridedBatched);

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

    template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRot_64)(hipblasHandle_t handle,
                                     int64_t         n,
                                     T1*             x,
                                     int64_t         incx,
                                     T1*             y,
                                     int64_t         incy,
                                     const T2*       c,
                                     const T3*       s);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotBatched_64)(hipblasHandle_t handle,
                                            int64_t         n,
                                            T1* const       x[],
                                            int64_t         incx,
                                            T1* const       y[],
                                            int64_t         incy,
                                            const T2*       c,
                                            const T3*       s,
                                            int64_t         batch_count);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotStridedBatched_64)(hipblasHandle_t handle,
                                                   int64_t         n,
                                                   T1*             x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   T1*             y,
                                                   int64_t         incy,
                                                   hipblasStride   stridey,
                                                   const T2*       c,
                                                   const T3*       s,
                                                   int64_t         batch_count);

    MAP2CF_D64(hipblasRot, float, float, float, hipblasSrot);
    MAP2CF_D64(hipblasRot, double, double, double, hipblasDrot);
    MAP2CF_D64(hipblasRot, hipComplex, float, hipComplex, hipblasCrot);
    MAP2CF_D64(hipblasRot, hipDoubleComplex, double, hipDoubleComplex, hipblasZrot);
    MAP2CF_D64(hipblasRot, hipComplex, float, float, hipblasCsrot);
    MAP2CF_D64(hipblasRot, hipDoubleComplex, double, double, hipblasZdrot);

    MAP2CF_D64(hipblasRotBatched, float, float, float, hipblasSrotBatched);
    MAP2CF_D64(hipblasRotBatched, double, double, double, hipblasDrotBatched);
    MAP2CF_D64(hipblasRotBatched, hipComplex, float, hipComplex, hipblasCrotBatched);
    MAP2CF_D64(
        hipblasRotBatched, hipDoubleComplex, double, hipDoubleComplex, hipblasZrotBatched);
    MAP2CF_D64(hipblasRotBatched, hipComplex, float, float, hipblasCsrotBatched);
    MAP2CF_D64(hipblasRotBatched, hipDoubleComplex, double, double, hipblasZdrotBatched);

    MAP2CF_D64(hipblasRotStridedBatched, float, float, float, hipblasSrotStridedBatched);
    MAP2CF_D64(hipblasRotStridedBatched, double, double, double, hipblasDrotStridedBatched);
    MAP2CF_D64(
        hipblasRotStridedBatched, hipComplex, float, hipComplex, hipblasCrotStridedBatched);
    MAP2CF_D64(hipblasRotStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipDoubleComplex,
                  hipblasZrotStridedBatched);
    MAP2CF_D64(
        hipblasRotStridedBatched, hipComplex, float, float, hipblasCsrotStridedBatched);
    MAP2CF_D64(
        hipblasRotStridedBatched, hipDoubleComplex, double, double, hipblasZdrotStridedBatched);

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

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotg_64)(hipblasHandle_t handle, T1* a, T1* b, T2* c, T1* s);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgBatched_64)(hipblasHandle_t handle,
                                             T1* const       a[],
                                             T1* const       b[],
                                             T2* const       c[],
                                             T1* const       s[],
                                             int64_t         batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgStridedBatched_64)(hipblasHandle_t handle,
                                                    T1*             a,
                                                    hipblasStride   stridea,
                                                    T1*             b,
                                                    hipblasStride   strideb,
                                                    T2*             c,
                                                    hipblasStride   stridec,
                                                    T1*             s,
                                                    hipblasStride   strides,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasRotg, float, float, hipblasSrotg);
    MAP2CF_D64(hipblasRotg, double, double, hipblasDrotg);
    MAP2CF_D64(hipblasRotg, hipComplex, float, hipblasCrotg);
    MAP2CF_D64(hipblasRotg, hipDoubleComplex, double, hipblasZrotg);

    MAP2CF_D64(hipblasRotgBatched, float, float, hipblasSrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, double, double, hipblasDrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, hipComplex, float, hipblasCrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, hipDoubleComplex, double, hipblasZrotgBatched);

    MAP2CF_D64(hipblasRotgStridedBatched, float, float, hipblasSrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched, double, double, hipblasDrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched, hipComplex, float, hipblasCrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZrotgStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotm_64)(
        hipblasHandle_t handle, int64_t n, T* x, int64_t incx, T* y, int64_t incy, const T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             T* const        x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             const T* const  param[],
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    T*              x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    const T*        param,
                                                    hipblasStride   strideparam,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasRotm, float, hipblasSrotm);
    MAP2CF_D64(hipblasRotm, double, hipblasDrotm);

    MAP2CF_D64(hipblasRotmBatched, float, hipblasSrotmBatched);
    MAP2CF_D64(hipblasRotmBatched, double, hipblasDrotmBatched);

    MAP2CF_D64(hipblasRotmStridedBatched, float, hipblasSrotmStridedBatched);
    MAP2CF_D64(hipblasRotmStridedBatched, double, hipblasDrotmStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmg_64)(
        hipblasHandle_t handle, T* d1, T* d2, T* x1, const T* y1, T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgBatched_64)(hipblasHandle_t handle,
                                              T* const        d1[],
                                              T* const        d2[],
                                              T* const        x1[],
                                              const T* const  y1[],
                                              T* const        param[],
                                              int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgStridedBatched_64)(hipblasHandle_t handle,
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
                                                     int64_t         batch_count);

    MAP2CF_D64(hipblasRotmg, float, hipblasSrotmg);
    MAP2CF_D64(hipblasRotmg, double, hipblasDrotmg);

    MAP2CF_D64(hipblasRotmgBatched, float, hipblasSrotmgBatched);
    MAP2CF_D64(hipblasRotmgBatched, double, hipblasDrotmgBatched);

    MAP2CF_D64(hipblasRotmgStridedBatched, float, hipblasSrotmgStridedBatched);
    MAP2CF_D64(hipblasRotmgStridedBatched, double, hipblasDrotmgStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamax_64)(
        hipblasHandle_t handle, int64_t n, const T* x, int64_t incx, int64_t* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxBatched_64)(hipblasHandle_t handle,
                                              int64_t         n,
                                              const T* const  x[],
                                              int64_t         incx,
                                              int64_t         batch_count,
                                              int64_t*        result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxStridedBatched_64)(hipblasHandle_t handle,
                                                     int64_t         n,
                                                     const T*        x,
                                                     int64_t         incx,
                                                     hipblasStride   stridex,
                                                     int64_t         batch_count,
                                                     int64_t*        result);

    MAP2CF_D64(hipblasIamax, float, hipblasIsamax);
    MAP2CF_D64(hipblasIamax, double, hipblasIdamax);
    MAP2CF_D64(hipblasIamax, hipComplex, hipblasIcamax);
    MAP2CF_D64(hipblasIamax, hipDoubleComplex, hipblasIzamax);

    MAP2CF_D64(hipblasIamaxBatched, float, hipblasIsamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, double, hipblasIdamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, hipComplex, hipblasIcamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, hipDoubleComplex, hipblasIzamaxBatched);

    MAP2CF_D64(hipblasIamaxStridedBatched, float, hipblasIsamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, double, hipblasIdamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, hipComplex, hipblasIcamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, hipDoubleComplex, hipblasIzamaxStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamin_64)(
        hipblasHandle_t handle, int64_t n, const T* x, int64_t incx, int64_t* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminBatched_64)(hipblasHandle_t handle,
                                              int64_t         n,
                                              const T* const  x[],
                                              int64_t         incx,
                                              int64_t         batch_count,
                                              int64_t*        result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminStridedBatched_64)(hipblasHandle_t handle,
                                                     int64_t         n,
                                                     const T*        x,
                                                     int64_t         incx,
                                                     hipblasStride   stridex,
                                                     int64_t         batch_count,
                                                     int64_t*        result);

    MAP2CF_D64(hipblasIamin, float, hipblasIsamin);
    MAP2CF_D64(hipblasIamin, double, hipblasIdamin);
    MAP2CF_D64(hipblasIamin, hipComplex, hipblasIcamin);
    MAP2CF_D64(hipblasIamin, hipDoubleComplex, hipblasIzamin);

    MAP2CF_D64(hipblasIaminBatched, float, hipblasIsaminBatched);
    MAP2CF_D64(hipblasIaminBatched, double, hipblasIdaminBatched);
    MAP2CF_D64(hipblasIaminBatched, hipComplex, hipblasIcaminBatched);
    MAP2CF_D64(hipblasIaminBatched, hipDoubleComplex, hipblasIzaminBatched);

    MAP2CF_D64(hipblasIaminStridedBatched, float, hipblasIsaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, double, hipblasIdaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, hipComplex, hipblasIcaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, hipDoubleComplex, hipblasIzaminStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpy_64)(hipblasHandle_t handle,
                                      int64_t         n,
                                      const T*        alpha,
                                      const T*        x,
                                      int64_t         incx,
                                      T*              y,
                                      int64_t         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T*        alpha,
                                             const T* const  x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T*        alpha,
                                                    const T*        x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasAxpy, hipblasHalf, hipblasHaxpy);
    MAP2CF_D64(hipblasAxpy, float, hipblasSaxpy);
    MAP2CF_D64(hipblasAxpy, double, hipblasDaxpy);
    MAP2CF_D64(hipblasAxpy, hipComplex, hipblasCaxpy);
    MAP2CF_D64(hipblasAxpy, hipDoubleComplex, hipblasZaxpy);

    MAP2CF_D64(hipblasAxpyBatched, hipblasHalf, hipblasHaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, float, hipblasSaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, double, hipblasDaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, hipComplex, hipblasCaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, hipDoubleComplex, hipblasZaxpyBatched);

    MAP2CF_D64(hipblasAxpyStridedBatched, hipblasHalf, hipblasHaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, float, hipblasSaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, double, hipblasDaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, hipComplex, hipblasCaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, hipDoubleComplex, hipblasZaxpyStridedBatched);

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

    // ger_64
    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGer_64)(hipblasHandle_t handle,
                                     int64_t         m,
                                     int64_t         n,
                                     const T*        alpha,
                                     const T*        x,
                                     int64_t         incx,
                                     const T*        y,
                                     int64_t         incy,
                                     T*              A,
                                     int64_t         lda);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerBatched_64)(hipblasHandle_t handle,
                                            int64_t         m,
                                            int64_t         n,
                                            const T*        alpha,
                                            const T* const  x[],
                                            int64_t         incx,
                                            const T* const  y[],
                                            int64_t         incy,
                                            T* const        A[],
                                            int64_t         lda,
                                            int64_t         batch_count);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerStridedBatched_64)(hipblasHandle_t handle,
                                                   int64_t         m,
                                                   int64_t         n,
                                                   const T*        alpha,
                                                   const T*        x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   const T*        y,
                                                   int64_t         incy,
                                                   hipblasStride   stridey,
                                                   T*              A,
                                                   int64_t         lda,
                                                   hipblasStride   strideA,
                                                   int64_t         batch_count);

    MAP2CF_D64(hipblasGer, float, false, hipblasSger);
    MAP2CF_D64(hipblasGer, double, false, hipblasDger);
    MAP2CF_D64(hipblasGer, hipComplex, false, hipblasCgeru);
    MAP2CF_D64(hipblasGer, hipDoubleComplex, false, hipblasZgeru);
    MAP2CF_D64(hipblasGer, hipComplex, true, hipblasCgerc);
    MAP2CF_D64(hipblasGer, hipDoubleComplex, true, hipblasZgerc);

    MAP2CF_D64(hipblasGerBatched, float, false, hipblasSgerBatched);
    MAP2CF_D64(hipblasGerBatched, double, false, hipblasDgerBatched);
    MAP2CF_D64(hipblasGerBatched, hipComplex, false, hipblasCgeruBatched);
    MAP2CF_D64(hipblasGerBatched, hipDoubleComplex, false, hipblasZgeruBatched);
    MAP2CF_D64(hipblasGerBatched, hipComplex, true, hipblasCgercBatched);
    MAP2CF_D64(hipblasGerBatched, hipDoubleComplex, true, hipblasZgercBatched);

    MAP2CF_D64(hipblasGerStridedBatched, float, false, hipblasSgerStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, double, false, hipblasDgerStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, hipComplex, false, hipblasCgeruStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched,
                  hipDoubleComplex,
                  false,
                  hipblasZgeruStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, hipComplex, true, hipblasCgercStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, hipDoubleComplex, true, hipblasZgercStridedBatched);

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
    hipblasStatus_t (*hipblasHbmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      int64_t           k,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    MAP2CF_D64(hipblasHbmv, hipComplex, hipblasChbmv);
    MAP2CF_D64(hipblasHbmv, hipDoubleComplex, hipblasZhbmv);

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
    hipblasStatus_t (*hipblasHbmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             int64_t           k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    MAP2CF_D64(hipblasHbmvBatched, hipComplex, hipblasChbmvBatched);
    MAP2CF_D64(hipblasHbmvBatched, hipDoubleComplex, hipblasZhbmvBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    int64_t           k,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasHbmvStridedBatched, hipComplex, hipblasChbmvStridedBatched);
    MAP2CF_D64(hipblasHbmvStridedBatched, hipDoubleComplex, hipblasZhbmvStridedBatched);

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
    hipblasStatus_t (*hipblasHemv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    MAP2CF_D64(hipblasHemv, hipComplex, hipblasChemv);
    MAP2CF_D64(hipblasHemv, hipDoubleComplex, hipblasZhemv);

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
    hipblasStatus_t (*hipblasHemvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batch_count);

    MAP2CF_D64(hipblasHemvBatched, hipComplex, hipblasChemvBatched);
    MAP2CF_D64(hipblasHemvBatched, hipDoubleComplex, hipblasZhemvBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     stride_a,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stride_x,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stride_y,
                                                    int64_t           batch_count);

    MAP2CF_D64(hipblasHemvStridedBatched, hipComplex, hipblasChemvStridedBatched);
    MAP2CF_D64(hipblasHemvStridedBatched, hipDoubleComplex, hipblasZhemvStridedBatched);

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
    hipblasStatus_t (*hipblasHer_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const U*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                A,
                                     int64_t           lda);

    MAP2CF_D64(hipblasHer, hipComplex, float, hipblasCher);
    MAP2CF_D64(hipblasHer, hipDoubleComplex, double, hipblasZher);

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
    hipblasStatus_t (*hipblasHerBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const U*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          A[],
                                            int64_t           lda,
                                            int64_t           batchCount);

    MAP2CF_D64(hipblasHerBatched, hipComplex, float, hipblasCherBatched);
    MAP2CF_D64(hipblasHerBatched, hipDoubleComplex, double, hipblasZherBatched);

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

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const U*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                A,
                                                   int64_t           lda,
                                                   hipblasStride     strideA,
                                                   int64_t           batchCount);

    MAP2CF_D64(hipblasHerStridedBatched, hipComplex, float, hipblasCherStridedBatched);
    MAP2CF_D64(hipblasHerStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZherStridedBatched);

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
    hipblasStatus_t (*hipblasHer2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                A,
                                      int64_t           lda);

    MAP2CF_D64(hipblasHer2, hipComplex, hipblasCher2);
    MAP2CF_D64(hipblasHer2, hipDoubleComplex, hipblasZher2);

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
    hipblasStatus_t (*hipblasHer2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          A[],
                                             int64_t           lda,
                                             int64_t           batchCount);

    MAP2CF_D64(hipblasHer2Batched, hipComplex, hipblasCher2Batched);
    MAP2CF_D64(hipblasHer2Batched, hipDoubleComplex, hipblasZher2Batched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasHer2StridedBatched, hipComplex, hipblasCher2StridedBatched);
    MAP2CF_D64(hipblasHer2StridedBatched, hipDoubleComplex, hipblasZher2StridedBatched);

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
    hipblasStatus_t (*hipblasHpmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          AP,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    MAP2CF_D64(hipblasHpmv, hipComplex, hipblasChpmv);
    MAP2CF_D64(hipblasHpmv, hipDoubleComplex, hipblasZhpmv);

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
    hipblasStatus_t (*hipblasHpmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    AP[],
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    MAP2CF_D64(hipblasHpmvBatched, hipComplex, hipblasChpmvBatched);
    MAP2CF_D64(hipblasHpmvBatched, hipDoubleComplex, hipblasZhpmvBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          AP,
                                                    hipblasStride     strideAP,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasHpmvStridedBatched, hipComplex, hipblasChpmvStridedBatched);
    MAP2CF_D64(hipblasHpmvStridedBatched, hipDoubleComplex, hipblasZhpmvStridedBatched);

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
    hipblasStatus_t (*hipblasHpr_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const U*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                AP);

    MAP2CF_D64(hipblasHpr, hipComplex, float, hipblasChpr);
    MAP2CF_D64(hipblasHpr, hipDoubleComplex, double, hipblasZhpr);

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
    hipblasStatus_t (*hipblasHprBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const U*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          AP[],
                                            int64_t           batchCount);

    MAP2CF_D64(hipblasHprBatched, hipComplex, float, hipblasChprBatched);
    MAP2CF_D64(hipblasHprBatched, hipDoubleComplex, double, hipblasZhprBatched);

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

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const U*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                AP,
                                                   hipblasStride     strideAP,
                                                   int64_t           batchCount);

    MAP2CF_D64(hipblasHprStridedBatched, hipComplex, float, hipblasChprStridedBatched);
    MAP2CF_D64(hipblasHprStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZhprStridedBatched);

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
    hipblasStatus_t (*hipblasHpr2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                AP);

    MAP2CF_D64(hipblasHpr2, hipComplex, hipblasChpr2);
    MAP2CF_D64(hipblasHpr2, hipDoubleComplex, hipblasZhpr2);

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
    hipblasStatus_t (*hipblasHpr2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          AP[],
                                             int64_t           batchCount);

    MAP2CF_D64(hipblasHpr2Batched, hipComplex, hipblasChpr2Batched);
    MAP2CF_D64(hipblasHpr2Batched, hipDoubleComplex, hipblasZhpr2Batched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                AP,
                                                    hipblasStride     strideAP,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasHpr2StridedBatched, hipComplex, hipblasChpr2StridedBatched);
    MAP2CF_D64(hipblasHpr2StridedBatched, hipDoubleComplex, hipblasZhpr2StridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      int64_t           k,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             int64_t           k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    int64_t           k,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSbmv, float, hipblasSsbmv);
    MAP2CF_D64(hipblasSbmv, double, hipblasDsbmv);

    MAP2CF_D64(hipblasSbmvBatched, float, hipblasSsbmvBatched);
    MAP2CF_D64(hipblasSbmvBatched, double, hipblasDsbmvBatched);

    MAP2CF_D64(hipblasSbmvStridedBatched, float, hipblasSsbmvStridedBatched);
    MAP2CF_D64(hipblasSbmvStridedBatched, double, hipblasDsbmvStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          AP,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    AP[],
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          AP,
                                                    hipblasStride     strideAP,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSpmv, float, hipblasSspmv);
    MAP2CF_D64(hipblasSpmv, double, hipblasDspmv);

    MAP2CF_D64(hipblasSpmvBatched, float, hipblasSspmvBatched);
    MAP2CF_D64(hipblasSpmvBatched, double, hipblasDspmvBatched);

    MAP2CF_D64(hipblasSpmvStridedBatched, float, hipblasSspmvStridedBatched);
    MAP2CF_D64(hipblasSpmvStridedBatched, double, hipblasDspmvStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const T*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const T*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          AP[],
                                            int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const T*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                AP,
                                                   hipblasStride     strideAP,
                                                   int64_t           batchCount);

    MAP2CF_D64(hipblasSpr, float, hipblasSspr);
    MAP2CF_D64(hipblasSpr, double, hipblasDspr);
    MAP2CF_D64(hipblasSpr, hipComplex, hipblasCspr);
    MAP2CF_D64(hipblasSpr, hipDoubleComplex, hipblasZspr);

    MAP2CF_D64(hipblasSprBatched, float, hipblasSsprBatched);
    MAP2CF_D64(hipblasSprBatched, double, hipblasDsprBatched);
    MAP2CF_D64(hipblasSprBatched, hipComplex, hipblasCsprBatched);
    MAP2CF_D64(hipblasSprBatched, hipDoubleComplex, hipblasZsprBatched);

    MAP2CF_D64(hipblasSprStridedBatched, float, hipblasSsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, double, hipblasDsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, hipComplex, hipblasCsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, hipDoubleComplex, hipblasZsprStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          AP[],
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                AP,
                                                    hipblasStride     strideAP,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSpr2, float, hipblasSspr2);
    MAP2CF_D64(hipblasSpr2, double, hipblasDspr2);

    MAP2CF_D64(hipblasSpr2Batched, float, hipblasSspr2Batched);
    MAP2CF_D64(hipblasSpr2Batched, double, hipblasDspr2Batched);

    MAP2CF_D64(hipblasSpr2StridedBatched, float, hipblasSspr2StridedBatched);
    MAP2CF_D64(hipblasSpr2StridedBatched, double, hipblasDspr2StridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSymv, float, hipblasSsymv);
    MAP2CF_D64(hipblasSymv, double, hipblasDsymv);
    MAP2CF_D64(hipblasSymv, hipComplex, hipblasCsymv);
    MAP2CF_D64(hipblasSymv, hipDoubleComplex, hipblasZsymv);

    MAP2CF_D64(hipblasSymvBatched, float, hipblasSsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, double, hipblasDsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, hipComplex, hipblasCsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, hipDoubleComplex, hipblasZsymvBatched);

    MAP2CF_D64(hipblasSymvStridedBatched, float, hipblasSsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, double, hipblasDsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, hipComplex, hipblasCsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, hipDoubleComplex, hipblasZsymvStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const T*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                A,
                                     int64_t           lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const T*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          A[],
                                            int64_t           lda,
                                            int64_t           batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const T*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                A,
                                                   int64_t           lda,
                                                   hipblasStride     strideA,
                                                   int64_t           batch_count);

    MAP2CF_D64(hipblasSyr, float, hipblasSsyr);
    MAP2CF_D64(hipblasSyr, double, hipblasDsyr);
    MAP2CF_D64(hipblasSyr, hipComplex, hipblasCsyr);
    MAP2CF_D64(hipblasSyr, hipDoubleComplex, hipblasZsyr);

    MAP2CF_D64(hipblasSyrBatched, float, hipblasSsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, double, hipblasDsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, hipComplex, hipblasCsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, hipDoubleComplex, hipblasZsyrBatched);

    MAP2CF_D64(hipblasSyrStridedBatched, float, hipblasSsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, double, hipblasDsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, hipComplex, hipblasCsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, hipDoubleComplex, hipblasZsyrStridedBatched);

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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                A,
                                      int64_t           lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          A[],
                                             int64_t           lda,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSyr2, float, hipblasSsyr2);
    MAP2CF_D64(hipblasSyr2, double, hipblasDsyr2);
    MAP2CF_D64(hipblasSyr2, hipComplex, hipblasCsyr2);
    MAP2CF_D64(hipblasSyr2, hipDoubleComplex, hipblasZsyr2);

    MAP2CF_D64(hipblasSyr2Batched, float, hipblasSsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, double, hipblasDsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, hipComplex, hipblasCsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, hipDoubleComplex, hipblasZsyr2Batched);

    MAP2CF_D64(hipblasSyr2StridedBatched, float, hipblasSsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, double, hipblasDsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, hipComplex, hipblasCsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, hipDoubleComplex, hipblasZsyr2StridedBatched);

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
    //tbmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            k,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            k,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            k,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTbmv, float, hipblasStbmv);
    MAP2CF_D64(hipblasTbmv, double, hipblasDtbmv);
    MAP2CF_D64(hipblasTbmv, hipComplex, hipblasCtbmv);
    MAP2CF_D64(hipblasTbmv, hipDoubleComplex, hipblasZtbmv);

    MAP2CF_D64(hipblasTbmvBatched, float, hipblasStbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, double, hipblasDtbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, hipComplex, hipblasCtbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, hipDoubleComplex, hipblasZtbmvBatched);

    MAP2CF_D64(hipblasTbmvStridedBatched, float, hipblasStbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, double, hipblasDtbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, hipComplex, hipblasCtbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, hipDoubleComplex, hipblasZtbmvStridedBatched);

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

    // tbsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            k,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            k,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            k,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTbsv, float, hipblasStbsv);
    MAP2CF_D64(hipblasTbsv, double, hipblasDtbsv);
    MAP2CF_D64(hipblasTbsv, hipComplex, hipblasCtbsv);
    MAP2CF_D64(hipblasTbsv, hipDoubleComplex, hipblasZtbsv);

    MAP2CF_D64(hipblasTbsvBatched, float, hipblasStbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, double, hipblasDtbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, hipComplex, hipblasCtbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, hipDoubleComplex, hipblasZtbsvBatched);

    MAP2CF_D64(hipblasTbsvStridedBatched, float, hipblasStbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, double, hipblasDtbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, hipComplex, hipblasCtbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, hipDoubleComplex, hipblasZtbsvStridedBatched);

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

    // tpmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           AP,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     AP[],
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           AP,
                                                    hipblasStride      strideAP,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTpmv, float, hipblasStpmv);
    MAP2CF_D64(hipblasTpmv, double, hipblasDtpmv);
    MAP2CF_D64(hipblasTpmv, hipComplex, hipblasCtpmv);
    MAP2CF_D64(hipblasTpmv, hipDoubleComplex, hipblasZtpmv);

    MAP2CF_D64(hipblasTpmvBatched, float, hipblasStpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, double, hipblasDtpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, hipComplex, hipblasCtpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, hipDoubleComplex, hipblasZtpmvBatched);

    MAP2CF_D64(hipblasTpmvStridedBatched, float, hipblasStpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, double, hipblasDtpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, hipComplex, hipblasCtpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, hipDoubleComplex, hipblasZtpmvStridedBatched);

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

    // tpsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           AP,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     AP[],
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           AP,
                                                    hipblasStride      strideAP,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTpsv, float, hipblasStpsv);
    MAP2CF_D64(hipblasTpsv, double, hipblasDtpsv);
    MAP2CF_D64(hipblasTpsv, hipComplex, hipblasCtpsv);
    MAP2CF_D64(hipblasTpsv, hipDoubleComplex, hipblasZtpsv);

    MAP2CF_D64(hipblasTpsvBatched, float, hipblasStpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, double, hipblasDtpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, hipComplex, hipblasCtpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, hipDoubleComplex, hipblasZtpsvBatched);

    MAP2CF_D64(hipblasTpsvStridedBatched, float, hipblasStpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, double, hipblasDtpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, hipComplex, hipblasCtpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, hipDoubleComplex, hipblasZtpsvStridedBatched);

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

    // trmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTrmv, float, hipblasStrmv);
    MAP2CF_D64(hipblasTrmv, double, hipblasDtrmv);
    MAP2CF_D64(hipblasTrmv, hipComplex, hipblasCtrmv);
    MAP2CF_D64(hipblasTrmv, hipDoubleComplex, hipblasZtrmv);

    MAP2CF_D64(hipblasTrmvBatched, float, hipblasStrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, double, hipblasDtrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, hipComplex, hipblasCtrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, hipDoubleComplex, hipblasZtrmvBatched);

    MAP2CF_D64(hipblasTrmvStridedBatched, float, hipblasStrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, double, hipblasDtrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, hipComplex, hipblasCtrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, hipDoubleComplex, hipblasZtrmvStridedBatched);

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

    // trsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTrsv, float, hipblasStrsv);
    MAP2CF_D64(hipblasTrsv, double, hipblasDtrsv);
    MAP2CF_D64(hipblasTrsv, hipComplex, hipblasCtrsv);
    MAP2CF_D64(hipblasTrsv, hipDoubleComplex, hipblasZtrsv);

    MAP2CF_D64(hipblasTrsvBatched, float, hipblasStrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, double, hipblasDtrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, hipComplex, hipblasCtrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, hipDoubleComplex, hipblasZtrsvBatched);

    MAP2CF_D64(hipblasTrsvStridedBatched, float, hipblasStrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, double, hipblasDtrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, hipComplex, hipblasCtrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, hipDoubleComplex, hipblasZtrsvStridedBatched);

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

    // gbmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmv_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      int64_t            m,
                                      int64_t            n,
                                      int64_t            kl,
                                      int64_t            ku,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           x,
                                      int64_t            incx,
                                      const T*           beta,
                                      T*                 y,
                                      int64_t            incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             int64_t            m,
                                             int64_t            n,
                                             int64_t            kl,
                                             int64_t            ku,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     x[],
                                             int64_t            incx,
                                             const T*           beta,
                                             T* const           y[],
                                             int64_t            incy,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    int64_t            kl,
                                                    int64_t            ku,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    const T*           x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    const T*           beta,
                                                    T*                 y,
                                                    int64_t            incy,
                                                    hipblasStride      stride_y,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasGbmv, float, hipblasSgbmv);
    MAP2CF_D64(hipblasGbmv, double, hipblasDgbmv);
    MAP2CF_D64(hipblasGbmv, hipComplex, hipblasCgbmv);
    MAP2CF_D64(hipblasGbmv, hipDoubleComplex, hipblasZgbmv);

    MAP2CF_D64(hipblasGbmvBatched, float, hipblasSgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, double, hipblasDgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, hipComplex, hipblasCgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, hipDoubleComplex, hipblasZgbmvBatched);

    MAP2CF_D64(hipblasGbmvStridedBatched, float, hipblasSgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, double, hipblasDgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, hipComplex, hipblasCgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, hipDoubleComplex, hipblasZgbmvStridedBatched);

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

    // gemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemv_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           x,
                                      int64_t            incx,
                                      const T*           beta,
                                      T*                 y,
                                      int64_t            incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     x[],
                                             int64_t            incx,
                                             const T*           beta,
                                             T* const           y[],
                                             int64_t            incy,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    const T*           x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    const T*           beta,
                                                    T*                 y,
                                                    int64_t            incy,
                                                    hipblasStride      stride_y,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasGemv, float, hipblasSgemv);
    MAP2CF_D64(hipblasGemv, double, hipblasDgemv);
    MAP2CF_D64(hipblasGemv, hipComplex, hipblasCgemv);
    MAP2CF_D64(hipblasGemv, hipDoubleComplex, hipblasZgemv);

    MAP2CF_D64(hipblasGemvBatched, float, hipblasSgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, double, hipblasDgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, hipComplex, hipblasCgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, hipDoubleComplex, hipblasZgemvBatched);

    MAP2CF_D64(hipblasGemvStridedBatched, float, hipblasSgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, double, hipblasDgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, hipComplex, hipblasCgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, hipDoubleComplex, hipblasZgemvStridedBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemm)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGemmStridedBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGemmBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemm_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      hipblasOperation_t transB,
                                      int64_t            m,
                                      int64_t            n,
                                      int64_t            k,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           B,
                                      int64_t            ldb,
                                      const T*           beta,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    hipblasOperation_t transB,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    int64_t            k,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    int64_t            bsa,
                                                    const T*           B,
                                                    int64_t            ldb,
                                                    int64_t            bsb,
                                                    const T*           beta,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    int64_t            bsc,
                                                    int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             hipblasOperation_t transB,
                                             int64_t            m,
                                             int64_t            n,
                                             int64_t            k,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     B[],
                                             int64_t            ldb,
                                             const T*           beta,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batch_count);

    MAP2CF_D64(hipblasGemm, hipblasHalf, hipblasHgemm);
    MAP2CF_D64(hipblasGemm, float, hipblasSgemm);
    MAP2CF_D64(hipblasGemm, double, hipblasDgemm);
    MAP2CF_D64(hipblasGemm, hipComplex, hipblasCgemm);
    MAP2CF_D64(hipblasGemm, hipDoubleComplex, hipblasZgemm);

    MAP2CF_D64(hipblasGemmBatched, hipblasHalf, hipblasHgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, float, hipblasSgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, double, hipblasDgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, hipComplex, hipblasCgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, hipDoubleComplex, hipblasZgemmBatched);

    MAP2CF_D64(hipblasGemmStridedBatched, hipblasHalf, hipblasHgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, float, hipblasSgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, double, hipblasDgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, hipComplex, hipblasCgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, hipDoubleComplex, hipblasZgemmStridedBatched);

    // herk
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerk)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasHerkBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasHerkStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerk_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      int64_t            n,
                                      int64_t            k,
                                      const U*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const U*           beta,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             int64_t            n,
                                             int64_t            k,
                                             const U*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const U*           beta,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    int64_t            n,
                                                    int64_t            k,
                                                    const U*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const U*           beta,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasHerk, hipComplex, float, hipblasCherk);
    MAP2CF_D64(hipblasHerk, hipDoubleComplex, double, hipblasZherk);

    MAP2CF_D64(hipblasHerkBatched, hipComplex, float, hipblasCherkBatched);
    MAP2CF_D64(hipblasHerkBatched, hipDoubleComplex, double, hipblasZherkBatched);

    MAP2CF_D64(hipblasHerkStridedBatched, hipComplex, float, hipblasCherkStridedBatched);
    MAP2CF_D64(hipblasHerkStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZherkStridedBatched);

    // her2k
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2k)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasHer2kBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasHer2kStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2k_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const U*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const U*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const U*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64(hipblasHer2k, hipComplex, float, hipblasCher2k);
    MAP2CF_D64(hipblasHer2k, hipDoubleComplex, double, hipblasZher2k);

    MAP2CF_D64(hipblasHer2kBatched, hipComplex, float, hipblasCher2kBatched);
    MAP2CF_D64(hipblasHer2kBatched, hipDoubleComplex, double, hipblasZher2kBatched);

    MAP2CF_D64(hipblasHer2kStridedBatched, hipComplex, float, hipblasCher2kStridedBatched);
    MAP2CF_D64(hipblasHer2kStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZher2kStridedBatched);

    // herkx
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkx)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasHerkxBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasHerkxStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkx_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const U*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const U*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const U*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64(hipblasHerkx, hipComplex, float, hipblasCherkx);
    MAP2CF_D64(hipblasHerkx, hipDoubleComplex, double, hipblasZherkx);

    MAP2CF_D64(hipblasHerkxBatched, hipComplex, float, hipblasCherkxBatched);
    MAP2CF_D64(hipblasHerkxBatched, hipDoubleComplex, double, hipblasZherkxBatched);

    MAP2CF_D64(hipblasHerkxStridedBatched, hipComplex, float, hipblasCherkxStridedBatched);
    MAP2CF_D64(hipblasHerkxStridedBatched,
                  hipDoubleComplex,
                  double,
                  hipblasZherkxStridedBatched);

    // symm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymm)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSymmBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasSymmStridedBatched)(hipblasHandle_t   handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymm_64)(hipblasHandle_t   handle,
                                      hipblasSideMode_t side,
                                      hipblasFillMode_t uplo,
                                      int64_t           m,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          B,
                                      int64_t           ldb,
                                      const T*          beta,
                                      T*                C,
                                      int64_t           ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmBatched_64)(hipblasHandle_t   handle,
                                             hipblasSideMode_t side,
                                             hipblasFillMode_t uplo,
                                             int64_t           m,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    B[],
                                             int64_t           ldb,
                                             const T*          beta,
                                             T* const          C[],
                                             int64_t           ldc,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasSideMode_t side,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           m,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          B,
                                                    int64_t           ldb,
                                                    hipblasStride     strideB,
                                                    const T*          beta,
                                                    T*                C,
                                                    int64_t           ldc,
                                                    hipblasStride     strideC,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSymm, float, hipblasSsymm);
    MAP2CF_D64(hipblasSymm, double, hipblasDsymm);
    MAP2CF_D64(hipblasSymm, hipComplex, hipblasCsymm);
    MAP2CF_D64(hipblasSymm, hipDoubleComplex, hipblasZsymm);

    MAP2CF_D64(hipblasSymmBatched, float, hipblasSsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, double, hipblasDsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, hipComplex, hipblasCsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, hipDoubleComplex, hipblasZsymmBatched);

    MAP2CF_D64(hipblasSymmStridedBatched, float, hipblasSsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, double, hipblasDsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, hipComplex, hipblasCsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, hipDoubleComplex, hipblasZsymmStridedBatched);

    // syrk
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrk)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasSyrkBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasSyrkStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrk_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      int64_t            n,
                                      int64_t            k,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           beta,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             int64_t            n,
                                             int64_t            k,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T*           beta,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    int64_t            n,
                                                    int64_t            k,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const T*           beta,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasSyrk, float, hipblasSsyrk);
    MAP2CF_D64(hipblasSyrk, double, hipblasDsyrk);
    MAP2CF_D64(hipblasSyrk, hipComplex, hipblasCsyrk);
    MAP2CF_D64(hipblasSyrk, hipDoubleComplex, hipblasZsyrk);

    MAP2CF_D64(hipblasSyrkBatched, float, hipblasSsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, double, hipblasDsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, hipComplex, hipblasCsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, hipDoubleComplex, hipblasZsyrkBatched);

    MAP2CF_D64(hipblasSyrkStridedBatched, float, hipblasSsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, double, hipblasDsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, hipComplex, hipblasCsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, hipDoubleComplex, hipblasZsyrkStridedBatched);

    // syr2k
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2k)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasSyr2kBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasSyr2kStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2k_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const T*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const T*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const T*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64(hipblasSyr2k, float, hipblasSsyr2k);
    MAP2CF_D64(hipblasSyr2k, double, hipblasDsyr2k);
    MAP2CF_D64(hipblasSyr2k, hipComplex, hipblasCsyr2k);
    MAP2CF_D64(hipblasSyr2k, hipDoubleComplex, hipblasZsyr2k);

    MAP2CF_D64(hipblasSyr2kBatched, float, hipblasSsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, double, hipblasDsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, hipComplex, hipblasCsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, hipDoubleComplex, hipblasZsyr2kBatched);

    MAP2CF_D64(hipblasSyr2kStridedBatched, float, hipblasSsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, double, hipblasDsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, hipComplex, hipblasCsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, hipDoubleComplex, hipblasZsyr2kStridedBatched);

    // syrkx
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkx)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasSyrkxBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasSyrkxStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkx_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const T*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const T*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const T*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64(hipblasSyrkx, float, hipblasSsyrkx);
    MAP2CF_D64(hipblasSyrkx, double, hipblasDsyrkx);
    MAP2CF_D64(hipblasSyrkx, hipComplex, hipblasCsyrkx);
    MAP2CF_D64(hipblasSyrkx, hipDoubleComplex, hipblasZsyrkx);

    MAP2CF_D64(hipblasSyrkxBatched, float, hipblasSsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, double, hipblasDsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, hipComplex, hipblasCsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, hipDoubleComplex, hipblasZsyrkxBatched);

    MAP2CF_D64(hipblasSyrkxStridedBatched, float, hipblasSsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, double, hipblasDsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, hipComplex, hipblasCsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, hipDoubleComplex, hipblasZsyrkxStridedBatched);

    // geam
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeam)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGeamBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGeamStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeam_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      hipblasOperation_t transB,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           beta,
                                      const T*           B,
                                      int64_t            ldb,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             hipblasOperation_t transB,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T*           beta,
                                             const T* const     B[],
                                             int64_t            ldb,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    hipblasOperation_t transB,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const T*           beta,
                                                    const T*           B,
                                                    int64_t            ldb,
                                                    hipblasStride      strideB,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasGeam, float, hipblasSgeam);
    MAP2CF_D64(hipblasGeam, double, hipblasDgeam);
    MAP2CF_D64(hipblasGeam, hipComplex, hipblasCgeam);
    MAP2CF_D64(hipblasGeam, hipDoubleComplex, hipblasZgeam);

    MAP2CF_D64(hipblasGeamBatched, float, hipblasSgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, double, hipblasDgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, hipComplex, hipblasCgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, hipDoubleComplex, hipblasZgeamBatched);

    MAP2CF_D64(hipblasGeamStridedBatched, float, hipblasSgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, double, hipblasDgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, hipComplex, hipblasCgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, hipDoubleComplex, hipblasZgeamStridedBatched);

    // hemm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemm)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHemmBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasHemmStridedBatched)(hipblasHandle_t   handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemm_64)(hipblasHandle_t   handle,
                                      hipblasSideMode_t side,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      int64_t           k,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          B,
                                      int64_t           ldb,
                                      const T*          beta,
                                      T*                C,
                                      int64_t           ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmBatched_64)(hipblasHandle_t   handle,
                                             hipblasSideMode_t side,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             int64_t           k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    B[],
                                             int64_t           ldb,
                                             const T*          beta,
                                             T* const          C[],
                                             int64_t           ldc,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasSideMode_t side,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    int64_t           k,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          B,
                                                    int64_t           ldb,
                                                    hipblasStride     strideB,
                                                    const T*          beta,
                                                    T*                C,
                                                    int64_t           ldc,
                                                    hipblasStride     strideC,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasHemm, hipComplex, hipblasChemm);
    MAP2CF_D64(hipblasHemm, hipDoubleComplex, hipblasZhemm);

    MAP2CF_D64(hipblasHemmBatched, hipComplex, hipblasChemmBatched);
    MAP2CF_D64(hipblasHemmBatched, hipDoubleComplex, hipblasZhemmBatched);

    MAP2CF_D64(hipblasHemmStridedBatched, hipComplex, hipblasChemmStridedBatched);
    MAP2CF_D64(hipblasHemmStridedBatched, hipDoubleComplex, hipblasZhemmStridedBatched);

    // trmm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmm)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTrmmBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTrmmStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmm_64)(hipblasHandle_t    handle,
                                      hipblasSideMode_t  side,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           B,
                                      int64_t            ldb,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmBatched_64)(hipblasHandle_t    handle,
                                             hipblasSideMode_t  side,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     B[],
                                             int64_t            ldb,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasSideMode_t  side,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const T*           B,
                                                    int64_t            ldb,
                                                    hipblasStride      strideB,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTrmm, float, hipblasStrmm);
    MAP2CF_D64(hipblasTrmm, double, hipblasDtrmm);
    MAP2CF_D64(hipblasTrmm, hipComplex, hipblasCtrmm);
    MAP2CF_D64(hipblasTrmm, hipDoubleComplex, hipblasZtrmm);

    MAP2CF_D64(hipblasTrmmBatched, float, hipblasStrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, double, hipblasDtrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, hipComplex, hipblasCtrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, hipDoubleComplex, hipblasZtrmmBatched);

    MAP2CF_D64(hipblasTrmmStridedBatched, float, hipblasStrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, double, hipblasDtrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, hipComplex, hipblasCtrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, hipDoubleComplex, hipblasZtrmmStridedBatched);

    // trsm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsm)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTrsmBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasTrsmStridedBatched)(hipblasHandle_t    handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsm_64)(hipblasHandle_t    handle,
                                      hipblasSideMode_t  side,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 B,
                                      int64_t            ldb);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmBatched_64)(hipblasHandle_t    handle,
                                             hipblasSideMode_t  side,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           B[],
                                             int64_t            ldb,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasSideMode_t  side,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    T*                 B,
                                                    int64_t            ldb,
                                                    hipblasStride      strideB,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTrsm, float, hipblasStrsm);
    MAP2CF_D64(hipblasTrsm, double, hipblasDtrsm);
    MAP2CF_D64(hipblasTrsm, hipComplex, hipblasCtrsm);
    MAP2CF_D64(hipblasTrsm, hipDoubleComplex, hipblasZtrsm);

    MAP2CF_D64(hipblasTrsmBatched, float, hipblasStrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, double, hipblasDtrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, hipComplex, hipblasCtrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, hipDoubleComplex, hipblasZtrsmBatched);

    MAP2CF_D64(hipblasTrsmStridedBatched, float, hipblasStrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, double, hipblasDtrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, hipComplex, hipblasCtrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, hipDoubleComplex, hipblasZtrsmStridedBatched);

    // dgmm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmm)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasDgmmBatched)(hipblasHandle_t   handle,
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
    hipblasStatus_t (*hipblasDgmmStridedBatched)(hipblasHandle_t   handle,
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

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmm_64)(hipblasHandle_t   handle,
                                      hipblasSideMode_t side,
                                      int64_t           m,
                                      int64_t           n,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      T*                C,
                                      int64_t           ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmBatched_64)(hipblasHandle_t   handle,
                                             hipblasSideMode_t side,
                                             int64_t           m,
                                             int64_t           n,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             T* const          C[],
                                             int64_t           ldc,
                                             int64_t           batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasSideMode_t side,
                                                    int64_t           m,
                                                    int64_t           n,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     stride_A,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stride_x,
                                                    T*                C,
                                                    int64_t           ldc,
                                                    hipblasStride     stride_C,
                                                    int64_t           batch_count);

    MAP2CF_D64(hipblasDgmm, float, hipblasSdgmm);
    MAP2CF_D64(hipblasDgmm, double, hipblasDdgmm);
    MAP2CF_D64(hipblasDgmm, hipComplex, hipblasCdgmm);
    MAP2CF_D64(hipblasDgmm, hipDoubleComplex, hipblasZdgmm);

    MAP2CF_D64(hipblasDgmmBatched, float, hipblasSdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, double, hipblasDdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, hipComplex, hipblasCdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, hipDoubleComplex, hipblasZdgmmBatched);

    MAP2CF_D64(hipblasDgmmStridedBatched, float, hipblasSdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, double, hipblasDdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, hipComplex, hipblasCdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, hipDoubleComplex, hipblasZdgmmStridedBatched);

    // trtri
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtri)(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    hipblasDiagType_t diag,
                                    int               n,
                                    const T*          A,
                                    int               lda,
                                    T*                invA,
                                    int               ldinvA);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtriBatched)(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           hipblasDiagType_t diag,
                                           int               n,
                                           const T* const    A[],
                                           int               lda,
                                           T*                invA[],
                                           int               ldinvA,
                                           int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtriStridedBatched)(hipblasHandle_t   handle,
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

    MAP2CF(hipblasTrtri, float, hipblasStrtri);
    MAP2CF(hipblasTrtri, double, hipblasDtrtri);
    MAP2CF(hipblasTrtri, hipComplex, hipblasCtrtri);
    MAP2CF(hipblasTrtri, hipDoubleComplex, hipblasZtrtri);

    MAP2CF(hipblasTrtriBatched, float, hipblasStrtriBatched);
    MAP2CF(hipblasTrtriBatched, double, hipblasDtrtriBatched);
    MAP2CF(hipblasTrtriBatched, hipComplex, hipblasCtrtriBatched);
    MAP2CF(hipblasTrtriBatched, hipDoubleComplex, hipblasZtrtriBatched);

    MAP2CF(hipblasTrtriStridedBatched, float, hipblasStrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, double, hipblasDtrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, hipComplex, hipblasCtrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, hipDoubleComplex, hipblasZtrtriStridedBatched);

#ifdef __HIP_PLATFORM_SOLVER__

    // getrf
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrf)(
        hipblasHandle_t handle, const int n, T* A, const int lda, int* ipiv, int* info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrfBatched)(hipblasHandle_t handle,
                                           const int       n,
                                           T* const        A[],
                                           const int       lda,
                                           int*            ipiv,
                                           int*            info,
                                           const int       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrfStridedBatched)(hipblasHandle_t     handle,
                                                  const int           n,
                                                  T*                  A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  int*                ipiv,
                                                  const hipblasStride strideP,
                                                  int*                info,
                                                  const int           batchCount);

    MAP2CF(hipblasGetrf, float, hipblasSgetrf);
    MAP2CF(hipblasGetrf, double, hipblasDgetrf);
    MAP2CF(hipblasGetrf, hipComplex, hipblasCgetrf);
    MAP2CF(hipblasGetrf, hipDoubleComplex, hipblasZgetrf);

    MAP2CF(hipblasGetrfBatched, float, hipblasSgetrfBatched);
    MAP2CF(hipblasGetrfBatched, double, hipblasDgetrfBatched);
    MAP2CF(hipblasGetrfBatched, hipComplex, hipblasCgetrfBatched);
    MAP2CF(hipblasGetrfBatched, hipDoubleComplex, hipblasZgetrfBatched);

    MAP2CF(hipblasGetrfStridedBatched, float, hipblasSgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, double, hipblasDgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, hipComplex, hipblasCgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, hipDoubleComplex, hipblasZgetrfStridedBatched);

    // getrs
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrs)(hipblasHandle_t          handle,
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
    hipblasStatus_t (*hipblasGetrsBatched)(hipblasHandle_t          handle,
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
    hipblasStatus_t (*hipblasGetrsStridedBatched)(hipblasHandle_t          handle,
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

    MAP2CF(hipblasGetrs, float, hipblasSgetrs);
    MAP2CF(hipblasGetrs, double, hipblasDgetrs);
    MAP2CF(hipblasGetrs, hipComplex, hipblasCgetrs);
    MAP2CF(hipblasGetrs, hipDoubleComplex, hipblasZgetrs);

    MAP2CF(hipblasGetrsBatched, float, hipblasSgetrsBatched);
    MAP2CF(hipblasGetrsBatched, double, hipblasDgetrsBatched);
    MAP2CF(hipblasGetrsBatched, hipComplex, hipblasCgetrsBatched);
    MAP2CF(hipblasGetrsBatched, hipDoubleComplex, hipblasZgetrsBatched);

    MAP2CF(hipblasGetrsStridedBatched, float, hipblasSgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, double, hipblasDgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, hipComplex, hipblasCgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, hipDoubleComplex, hipblasZgetrsStridedBatched);

    // getri
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetriBatched)(hipblasHandle_t handle,
                                           const int       n,
                                           T* const        A[],
                                           const int       lda,
                                           int*            ipiv,
                                           T* const        C[],
                                           const int       ldc,
                                           int*            info,
                                           const int       batchCount);

    MAP2CF(hipblasGetriBatched, float, hipblasSgetriBatched);
    MAP2CF(hipblasGetriBatched, double, hipblasDgetriBatched);
    MAP2CF(hipblasGetriBatched, hipComplex, hipblasCgetriBatched);
    MAP2CF(hipblasGetriBatched, hipDoubleComplex, hipblasZgetriBatched);

    // geqrf
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrf)(
        hipblasHandle_t handle, const int m, const int n, T* A, const int lda, T* ipiv, int* info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrfBatched)(hipblasHandle_t handle,
                                           const int       m,
                                           const int       n,
                                           T* const        A[],
                                           const int       lda,
                                           T* const        ipiv[],
                                           int*            info,
                                           const int       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrfStridedBatched)(hipblasHandle_t     handle,
                                                  const int           m,
                                                  const int           n,
                                                  T*                  A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  T*                  ipiv,
                                                  const hipblasStride strideP,
                                                  int*                info,
                                                  const int           batchCount);

    MAP2CF(hipblasGeqrf, float, hipblasSgeqrf);
    MAP2CF(hipblasGeqrf, double, hipblasDgeqrf);
    MAP2CF(hipblasGeqrf, hipComplex, hipblasCgeqrf);
    MAP2CF(hipblasGeqrf, hipDoubleComplex, hipblasZgeqrf);

    MAP2CF(hipblasGeqrfBatched, float, hipblasSgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, double, hipblasDgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, hipComplex, hipblasCgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, hipDoubleComplex, hipblasZgeqrfBatched);

    MAP2CF(hipblasGeqrfStridedBatched, float, hipblasSgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, double, hipblasDgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, hipComplex, hipblasCgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, hipDoubleComplex, hipblasZgeqrfStridedBatched);

    // gels
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGels)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGelsBatched)(hipblasHandle_t    handle,
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
    hipblasStatus_t (*hipblasGelsStridedBatched)(hipblasHandle_t     handle,
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

    MAP2CF(hipblasGels, float, hipblasSgels);
    MAP2CF(hipblasGels, double, hipblasDgels);
    MAP2CF(hipblasGels, hipComplex, hipblasCgels);
    MAP2CF(hipblasGels, hipDoubleComplex, hipblasZgels);

    MAP2CF(hipblasGelsBatched, float, hipblasSgelsBatched);
    MAP2CF(hipblasGelsBatched, double, hipblasDgelsBatched);
    MAP2CF(hipblasGelsBatched, hipComplex, hipblasCgelsBatched);
    MAP2CF(hipblasGelsBatched, hipDoubleComplex, hipblasZgelsBatched);

    MAP2CF(hipblasGelsStridedBatched, float, hipblasSgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, double, hipblasDgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, hipComplex, hipblasCgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, hipDoubleComplex, hipblasZgelsStridedBatched);

#endif
}

#endif // _ROCBLAS_HPP_
