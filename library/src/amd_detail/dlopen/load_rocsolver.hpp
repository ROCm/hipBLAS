/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#if BUILD_WITH_SOLVER

#include <rocsolver/rocsolver.h>

#else

#include "rocblas/rocblas.h"
// function pointer declarations

// getrf
using fp_rocsolver_sgetrf      = rocblas_status (*)(rocblas_handle,
                                               const rocblas_int,
                                               const rocblas_int,
                                               float*,
                                               const rocblas_int,
                                               rocblas_int*,
                                               rocblas_int*);
using fp_rocsolver_dgetrf      = rocblas_status (*)(rocblas_handle,
                                               const rocblas_int,
                                               const rocblas_int,
                                               double*,
                                               const rocblas_int,
                                               rocblas_int*,
                                               rocblas_int*);
using fp_rocsolver_cgetrf      = rocblas_status (*)(rocblas_handle,
                                               const rocblas_int,
                                               const rocblas_int,
                                               rocblas_float_complex*,
                                               const rocblas_int,
                                               rocblas_int*,
                                               rocblas_int*);
using fp_rocsolver_zgetrf      = rocblas_status (*)(rocblas_handle,
                                               const rocblas_int,
                                               const rocblas_int,
                                               rocblas_double_complex*,
                                               const rocblas_int,
                                               rocblas_int*,
                                               rocblas_int*);
using fp_rocsolver_sgetrf_npvt = rocblas_status (*)(
    rocblas_handle, const rocblas_int, const rocblas_int, float*, rocblas_int, rocblas_int*);
using fp_rocsolver_dgetrf_npvt = rocblas_status (*)(
    rocblas_handle, const rocblas_int, const rocblas_int, double*, rocblas_int, rocblas_int*);
using fp_rocsolver_cgetrf_npvt = rocblas_status (*)(rocblas_handle,
                                                    const rocblas_int,
                                                    const rocblas_int,
                                                    rocblas_float_complex*,
                                                    rocblas_int,
                                                    rocblas_int*);
using fp_rocsolver_zgetrf_npvt = rocblas_status (*)(rocblas_handle,
                                                    const rocblas_int,
                                                    const rocblas_int,
                                                    rocblas_double_complex*,
                                                    rocblas_int,
                                                    rocblas_int*);

// getrf_batched
using fp_rocsolver_sgetrf_batched      = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       float* const*,
                                                       const rocblas_int,
                                                       rocblas_int*,
                                                       const rocblas_stride,
                                                       rocblas_int*,
                                                       const rocblas_int);
using fp_rocsolver_dgetrf_batched      = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       double* const*,
                                                       const rocblas_int,
                                                       rocblas_int*,
                                                       const rocblas_stride,
                                                       rocblas_int*,
                                                       const rocblas_int);
using fp_rocsolver_cgetrf_batched      = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       rocblas_float_complex* const*,
                                                       const rocblas_int,
                                                       rocblas_int*,
                                                       const rocblas_stride,
                                                       rocblas_int*,
                                                       const rocblas_int);
using fp_rocsolver_zgetrf_batched      = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       rocblas_double_complex* const*,
                                                       const rocblas_int,
                                                       rocblas_int*,
                                                       const rocblas_stride,
                                                       rocblas_int*,
                                                       const rocblas_int);
using fp_rocsolver_sgetrf_npvt_batched = rocblas_status (*)(rocblas_handle,
                                                            const rocblas_int,
                                                            const rocblas_int,
                                                            float* const*,
                                                            const rocblas_int,
                                                            rocblas_int*,
                                                            const rocblas_int);
using fp_rocsolver_dgetrf_npvt_batched = rocblas_status (*)(rocblas_handle,
                                                            const rocblas_int,
                                                            const rocblas_int,
                                                            double* const*,
                                                            const rocblas_int,
                                                            rocblas_int*,
                                                            const rocblas_int);
using fp_rocsolver_cgetrf_npvt_batched = rocblas_status (*)(rocblas_handle,
                                                            const rocblas_int,
                                                            const rocblas_int,
                                                            rocblas_float_complex* const*,
                                                            const rocblas_int,
                                                            rocblas_int*,
                                                            const rocblas_int);
using fp_rocsolver_zgetrf_npvt_batched = rocblas_status (*)(rocblas_handle,
                                                            const rocblas_int,
                                                            const rocblas_int,
                                                            rocblas_double_complex* const*,
                                                            const rocblas_int,
                                                            rocblas_int*,
                                                            const rocblas_int);

// getrf_strided_batched
using fp_rocsolver_sgetrf_strided_batched      = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               float*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_int);
using fp_rocsolver_dgetrf_strided_batched      = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               double*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_int);
using fp_rocsolver_cgetrf_strided_batched      = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               rocblas_float_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_int);
using fp_rocsolver_zgetrf_strided_batched      = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               rocblas_double_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_stride,
                                                               rocblas_int*,
                                                               const rocblas_int);
using fp_rocsolver_sgetrf_npvt_strided_batched = rocblas_status (*)(rocblas_handle,
                                                                    const rocblas_int,
                                                                    const rocblas_int,
                                                                    float*,
                                                                    const rocblas_int,
                                                                    const rocblas_stride,
                                                                    rocblas_int*,
                                                                    const rocblas_int);
using fp_rocsolver_dgetrf_npvt_strided_batched = rocblas_status (*)(rocblas_handle,
                                                                    const rocblas_int,
                                                                    const rocblas_int,
                                                                    double*,
                                                                    const rocblas_int,
                                                                    const rocblas_stride,
                                                                    rocblas_int*,
                                                                    const rocblas_int);
using fp_rocsolver_cgetrf_npvt_strided_batched = rocblas_status (*)(rocblas_handle,
                                                                    const rocblas_int,
                                                                    const rocblas_int,
                                                                    rocblas_float_complex*,
                                                                    const rocblas_int,
                                                                    const rocblas_stride,
                                                                    rocblas_int*,
                                                                    const rocblas_int);
using fp_rocsolver_zgetrf_npvt_strided_batched = rocblas_status (*)(rocblas_handle,
                                                                    const rocblas_int,
                                                                    const rocblas_int,
                                                                    rocblas_double_complex*,
                                                                    const rocblas_int,
                                                                    const rocblas_stride,
                                                                    rocblas_int*,
                                                                    const rocblas_int);

// getrs
using fp_rocsolver_sgetrs = rocblas_status (*)(rocblas_handle,
                                               const rocblas_operation,
                                               const rocblas_int,
                                               const rocblas_int,
                                               float*,
                                               const rocblas_int,
                                               const rocblas_int*,
                                               float*,
                                               const rocblas_int);
using fp_rocsolver_dgetrs = rocblas_status (*)(rocblas_handle,
                                               const rocblas_operation,
                                               const rocblas_int,
                                               const rocblas_int,
                                               double*,
                                               const rocblas_int,
                                               const rocblas_int*,
                                               double*,
                                               const rocblas_int);
using fp_rocsolver_cgetrs = rocblas_status (*)(rocblas_handle,
                                               const rocblas_operation,
                                               const rocblas_int,
                                               const rocblas_int,
                                               rocblas_float_complex*,
                                               const rocblas_int,
                                               const rocblas_int*,
                                               rocblas_float_complex*,
                                               const rocblas_int);
using fp_rocsolver_zgetrs = rocblas_status (*)(rocblas_handle,
                                               const rocblas_operation,
                                               const rocblas_int,
                                               const rocblas_int,
                                               rocblas_double_complex*,
                                               const rocblas_int,
                                               const rocblas_int*,
                                               rocblas_double_complex*,
                                               const rocblas_int);

using fp_rocsolver_sgetrs_batched = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_operation,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       float* const*,
                                                       const rocblas_int,
                                                       const rocblas_int*,
                                                       const rocblas_stride,
                                                       float* const*,
                                                       const rocblas_int,
                                                       const rocblas_int);
using fp_rocsolver_dgetrs_batched = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_operation,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       double* const*,
                                                       const rocblas_int,
                                                       const rocblas_int*,
                                                       const rocblas_stride,
                                                       double* const*,
                                                       const rocblas_int,
                                                       const rocblas_int);
using fp_rocsolver_cgetrs_batched = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_operation,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       rocblas_float_complex* const*,
                                                       const rocblas_int,
                                                       const rocblas_int*,
                                                       const rocblas_stride,
                                                       rocblas_float_complex* const*,
                                                       const rocblas_int,
                                                       const rocblas_int);
using fp_rocsolver_zgetrs_batched = rocblas_status (*)(rocblas_handle,
                                                       const rocblas_operation,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       rocblas_double_complex* const*,
                                                       const rocblas_int,
                                                       const rocblas_int*,
                                                       const rocblas_stride,
                                                       rocblas_double_complex* const*,
                                                       const rocblas_int,
                                                       const rocblas_int);

using fp_rocsolver_sgetrs_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_operation,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               float*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int*,
                                                               const rocblas_stride,
                                                               float*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int);
using fp_rocsolver_dgetrs_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_operation,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               double*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int*,
                                                               const rocblas_stride,
                                                               double*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int);
using fp_rocsolver_cgetrs_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_operation,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               rocblas_float_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int*,
                                                               const rocblas_stride,
                                                               rocblas_float_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int);
using fp_rocsolver_zgetrs_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_operation,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               rocblas_double_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int*,
                                                               const rocblas_stride,
                                                               rocblas_double_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               const rocblas_int);

// getri_batched
using fp_rocsolver_sgetri_outofplace_batched = rocblas_status (*)(rocblas_handle,
                                                                  const rocblas_int,
                                                                  float* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_stride,
                                                                  float* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_int);
using fp_rocsolver_dgetri_outofplace_batched = rocblas_status (*)(rocblas_handle,
                                                                  const rocblas_int,
                                                                  double* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_stride,
                                                                  double* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_int);
using fp_rocsolver_cgetri_outofplace_batched = rocblas_status (*)(rocblas_handle,
                                                                  const rocblas_int,
                                                                  rocblas_float_complex* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_stride,
                                                                  rocblas_float_complex* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_int);
using fp_rocsolver_zgetri_outofplace_batched = rocblas_status (*)(rocblas_handle,
                                                                  const rocblas_int,
                                                                  rocblas_double_complex* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_stride,
                                                                  rocblas_double_complex* const*,
                                                                  const rocblas_int,
                                                                  rocblas_int*,
                                                                  const rocblas_int);

using fp_rocsolver_sgetri_npvt_outofplace_batched = rocblas_status (*)(rocblas_handle,
                                                                       const rocblas_int,
                                                                       float* const*,
                                                                       const rocblas_int,
                                                                       float* const*,
                                                                       const rocblas_int,
                                                                       rocblas_int*,
                                                                       const rocblas_int);
using fp_rocsolver_dgetri_npvt_outofplace_batched = rocblas_status (*)(rocblas_handle,
                                                                       const rocblas_int,
                                                                       double* const*,
                                                                       const rocblas_int,
                                                                       double* const*,
                                                                       const rocblas_int,
                                                                       rocblas_int*,
                                                                       const rocblas_int);
using fp_rocsolver_cgetri_npvt_outofplace_batched
    = rocblas_status (*)(rocblas_handle,
                         const rocblas_int,
                         rocblas_float_complex* const*,
                         const rocblas_int,
                         rocblas_float_complex* const*,
                         const rocblas_int,
                         rocblas_int*,
                         const rocblas_int);
using fp_rocsolver_zgetri_npvt_outofplace_batched
    = rocblas_status (*)(rocblas_handle,
                         const rocblas_int,
                         rocblas_double_complex* const*,
                         const rocblas_int,
                         rocblas_double_complex* const*,
                         const rocblas_int,
                         rocblas_int*,
                         const rocblas_int);

// geqrf
using fp_rocsolver_sgeqrf = rocblas_status (*)(
    rocblas_handle, const rocblas_int, const rocblas_int, float*, const rocblas_int, float*);
using fp_rocsolver_dgeqrf = rocblas_status (*)(
    rocblas_handle, const rocblas_int, const rocblas_int, double*, const rocblas_int, double*);
using fp_rocsolver_cgeqrf = rocblas_status (*)(rocblas_handle,
                                               const rocblas_int,
                                               const rocblas_int,
                                               rocblas_float_complex*,
                                               const rocblas_int,
                                               rocblas_float_complex*);
using fp_rocsolver_zgeqrf = rocblas_status (*)(rocblas_handle,
                                               const rocblas_int,
                                               const rocblas_int,
                                               rocblas_double_complex*,
                                               const rocblas_int,
                                               rocblas_double_complex*);

using fp_rocsolver_sgeqrf_ptr_batched = rocblas_status (*)(rocblas_handle,
                                                           const rocblas_int,
                                                           const rocblas_int,
                                                           float* const*,
                                                           const rocblas_int,
                                                           float* const*,
                                                           const rocblas_int);
using fp_rocsolver_dgeqrf_ptr_batched = rocblas_status (*)(rocblas_handle,
                                                           const rocblas_int,
                                                           const rocblas_int,
                                                           double* const*,
                                                           const rocblas_int,
                                                           double* const*,
                                                           const rocblas_int);
using fp_rocsolver_cgeqrf_ptr_batched = rocblas_status (*)(rocblas_handle,
                                                           const rocblas_int,
                                                           const rocblas_int,
                                                           rocblas_float_complex* const*,
                                                           const rocblas_int,
                                                           rocblas_float_complex* const*,
                                                           const rocblas_int);
using fp_rocsolver_zgeqrf_ptr_batched = rocblas_status (*)(rocblas_handle,
                                                           const rocblas_int,
                                                           const rocblas_int,
                                                           rocblas_double_complex* const*,
                                                           const rocblas_int,
                                                           rocblas_double_complex* const*,
                                                           const rocblas_int);

using fp_rocsolver_sgeqrf_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               float*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               float*,
                                                               const rocblas_stride,
                                                               const rocblas_int);
using fp_rocsolver_dgeqrf_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               double*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               double*,
                                                               const rocblas_stride,
                                                               const rocblas_int);
using fp_rocsolver_cgeqrf_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               rocblas_float_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               rocblas_float_complex*,
                                                               const rocblas_stride,
                                                               const rocblas_int);
using fp_rocsolver_zgeqrf_strided_batched = rocblas_status (*)(rocblas_handle,
                                                               const rocblas_int,
                                                               const rocblas_int,
                                                               rocblas_double_complex*,
                                                               const rocblas_int,
                                                               const rocblas_stride,
                                                               rocblas_double_complex*,
                                                               const rocblas_stride,
                                                               const rocblas_int);

// gels
using fp_rocsolver_sgels = rocblas_status (*)(rocblas_handle,
                                              rocblas_operation,
                                              const rocblas_int,
                                              const rocblas_int,
                                              const rocblas_int,
                                              float*,
                                              const rocblas_int,
                                              float*,
                                              const rocblas_int,
                                              rocblas_int*);
using fp_rocsolver_dgels = rocblas_status (*)(rocblas_handle,
                                              rocblas_operation,
                                              const rocblas_int,
                                              const rocblas_int,
                                              const rocblas_int,
                                              double*,
                                              const rocblas_int,
                                              double*,
                                              const rocblas_int,
                                              rocblas_int*);
using fp_rocsolver_cgels = rocblas_status (*)(rocblas_handle,
                                              rocblas_operation,
                                              const rocblas_int,
                                              const rocblas_int,
                                              const rocblas_int,
                                              rocblas_float_complex*,
                                              const rocblas_int,
                                              rocblas_float_complex*,
                                              const rocblas_int,
                                              rocblas_int*);
using fp_rocsolver_zgels = rocblas_status (*)(rocblas_handle,
                                              rocblas_operation,
                                              const rocblas_int,
                                              const rocblas_int,
                                              const rocblas_int,
                                              rocblas_double_complex*,
                                              const rocblas_int,
                                              rocblas_double_complex*,
                                              const rocblas_int,
                                              rocblas_int*);

using fp_rocsolver_sgels_batched = rocblas_status (*)(rocblas_handle,
                                                      rocblas_operation,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      float* const*,
                                                      const rocblas_int,
                                                      float* const*,
                                                      const rocblas_int,
                                                      rocblas_int*,
                                                      const rocblas_int);
using fp_rocsolver_dgels_batched = rocblas_status (*)(rocblas_handle,
                                                      rocblas_operation,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      double* const*,
                                                      const rocblas_int,
                                                      double* const*,
                                                      const rocblas_int,
                                                      rocblas_int*,
                                                      const rocblas_int);
using fp_rocsolver_cgels_batched = rocblas_status (*)(rocblas_handle,
                                                      rocblas_operation,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      rocblas_float_complex* const*,
                                                      const rocblas_int,
                                                      rocblas_float_complex* const*,
                                                      const rocblas_int,
                                                      rocblas_int*,
                                                      const rocblas_int);
using fp_rocsolver_zgels_batched = rocblas_status (*)(rocblas_handle,
                                                      rocblas_operation,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      const rocblas_int,
                                                      rocblas_double_complex* const*,
                                                      const rocblas_int,
                                                      rocblas_double_complex* const*,
                                                      const rocblas_int,
                                                      rocblas_int*,
                                                      const rocblas_int);

using fp_rocsolver_sgels_strided_batched = rocblas_status (*)(rocblas_handle,
                                                              rocblas_operation,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              float*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              float*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_int*,
                                                              const rocblas_int);
using fp_rocsolver_dgels_strided_batched = rocblas_status (*)(rocblas_handle,
                                                              rocblas_operation,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              double*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              double*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_int*,
                                                              const rocblas_int);
using fp_rocsolver_cgels_strided_batched = rocblas_status (*)(rocblas_handle,
                                                              rocblas_operation,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              rocblas_float_complex*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_float_complex*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_int*,
                                                              const rocblas_int);
using fp_rocsolver_zgels_strided_batched = rocblas_status (*)(rocblas_handle,
                                                              rocblas_operation,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              rocblas_double_complex*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_double_complex*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_int*,
                                                              const rocblas_int);

extern fp_rocsolver_sgetrf                 g_rocsolver_sgetrf; // our global function pointer
extern fp_rocsolver_dgetrf                 g_rocsolver_dgetrf;
extern fp_rocsolver_cgetrf                 g_rocsolver_cgetrf;
extern fp_rocsolver_zgetrf                 g_rocsolver_zgetrf;
extern fp_rocsolver_sgetrf_batched         g_rocsolver_sgetrf_batched;
extern fp_rocsolver_dgetrf_batched         g_rocsolver_dgetrf_batched;
extern fp_rocsolver_cgetrf_batched         g_rocsolver_cgetrf_batched;
extern fp_rocsolver_zgetrf_batched         g_rocsolver_zgetrf_batched;
extern fp_rocsolver_sgetrf_strided_batched g_rocsolver_sgetrf_strided_batched;
extern fp_rocsolver_dgetrf_strided_batched g_rocsolver_dgetrf_strided_batched;
extern fp_rocsolver_cgetrf_strided_batched g_rocsolver_cgetrf_strided_batched;
extern fp_rocsolver_zgetrf_strided_batched g_rocsolver_zgetrf_strided_batched;

extern fp_rocsolver_sgetrf_npvt                 g_rocsolver_sgetrf_npvt;
extern fp_rocsolver_dgetrf_npvt                 g_rocsolver_dgetrf_npvt;
extern fp_rocsolver_cgetrf_npvt                 g_rocsolver_cgetrf_npvt;
extern fp_rocsolver_zgetrf_npvt                 g_rocsolver_zgetrf_npvt;
extern fp_rocsolver_sgetrf_npvt_batched         g_rocsolver_sgetrf_npvt_batched;
extern fp_rocsolver_dgetrf_npvt_batched         g_rocsolver_dgetrf_npvt_batched;
extern fp_rocsolver_cgetrf_npvt_batched         g_rocsolver_cgetrf_npvt_batched;
extern fp_rocsolver_zgetrf_npvt_batched         g_rocsolver_zgetrf_npvt_batched;
extern fp_rocsolver_sgetrf_npvt_strided_batched g_rocsolver_sgetrf_npvt_strided_batched;
extern fp_rocsolver_dgetrf_npvt_strided_batched g_rocsolver_dgetrf_npvt_strided_batched;
extern fp_rocsolver_cgetrf_npvt_strided_batched g_rocsolver_cgetrf_npvt_strided_batched;
extern fp_rocsolver_zgetrf_npvt_strided_batched g_rocsolver_zgetrf_npvt_strided_batched;

extern fp_rocsolver_sgetrs                 g_rocsolver_sgetrs;
extern fp_rocsolver_dgetrs                 g_rocsolver_dgetrs;
extern fp_rocsolver_cgetrs                 g_rocsolver_cgetrs;
extern fp_rocsolver_zgetrs                 g_rocsolver_zgetrs;
extern fp_rocsolver_sgetrs_batched         g_rocsolver_sgetrs_batched;
extern fp_rocsolver_dgetrs_batched         g_rocsolver_dgetrs_batched;
extern fp_rocsolver_cgetrs_batched         g_rocsolver_cgetrs_batched;
extern fp_rocsolver_zgetrs_batched         g_rocsolver_zgetrs_batched;
extern fp_rocsolver_sgetrs_strided_batched g_rocsolver_sgetrs_strided_batched;
extern fp_rocsolver_dgetrs_strided_batched g_rocsolver_dgetrs_strided_batched;
extern fp_rocsolver_cgetrs_strided_batched g_rocsolver_cgetrs_strided_batched;
extern fp_rocsolver_zgetrs_strided_batched g_rocsolver_zgetrs_strided_batched;

extern fp_rocsolver_sgetri_outofplace_batched      g_rocsolver_sgetri_outofplace_batched;
extern fp_rocsolver_dgetri_outofplace_batched      g_rocsolver_dgetri_outofplace_batched;
extern fp_rocsolver_cgetri_outofplace_batched      g_rocsolver_cgetri_outofplace_batched;
extern fp_rocsolver_zgetri_outofplace_batched      g_rocsolver_zgetri_outofplace_batched;
extern fp_rocsolver_sgetri_npvt_outofplace_batched g_rocsolver_sgetri_npvt_outofplace_batched;
extern fp_rocsolver_dgetri_npvt_outofplace_batched g_rocsolver_dgetri_npvt_outofplace_batched;
extern fp_rocsolver_cgetri_npvt_outofplace_batched g_rocsolver_cgetri_npvt_outofplace_batched;
extern fp_rocsolver_zgetri_npvt_outofplace_batched g_rocsolver_zgetri_npvt_outofplace_batched;

extern fp_rocsolver_sgeqrf                 g_rocsolver_sgeqrf;
extern fp_rocsolver_dgeqrf                 g_rocsolver_dgeqrf;
extern fp_rocsolver_cgeqrf                 g_rocsolver_cgeqrf;
extern fp_rocsolver_zgeqrf                 g_rocsolver_zgeqrf;
extern fp_rocsolver_sgeqrf_ptr_batched     g_rocsolver_sgeqrf_ptr_batched;
extern fp_rocsolver_dgeqrf_ptr_batched     g_rocsolver_dgeqrf_ptr_batched;
extern fp_rocsolver_cgeqrf_ptr_batched     g_rocsolver_cgeqrf_ptr_batched;
extern fp_rocsolver_zgeqrf_ptr_batched     g_rocsolver_zgeqrf_ptr_batched;
extern fp_rocsolver_sgeqrf_strided_batched g_rocsolver_sgeqrf_strided_batched;
extern fp_rocsolver_dgeqrf_strided_batched g_rocsolver_dgeqrf_strided_batched;
extern fp_rocsolver_cgeqrf_strided_batched g_rocsolver_cgeqrf_strided_batched;
extern fp_rocsolver_zgeqrf_strided_batched g_rocsolver_zgeqrf_strided_batched;

extern fp_rocsolver_sgels                 g_rocsolver_sgels;
extern fp_rocsolver_dgels                 g_rocsolver_dgels;
extern fp_rocsolver_cgels                 g_rocsolver_cgels;
extern fp_rocsolver_zgels                 g_rocsolver_zgels;
extern fp_rocsolver_sgels_batched         g_rocsolver_sgels_batched;
extern fp_rocsolver_dgels_batched         g_rocsolver_dgels_batched;
extern fp_rocsolver_cgels_batched         g_rocsolver_cgels_batched;
extern fp_rocsolver_zgels_batched         g_rocsolver_zgels_batched;
extern fp_rocsolver_sgels_strided_batched g_rocsolver_sgels_strided_batched;
extern fp_rocsolver_dgels_strided_batched g_rocsolver_dgels_strided_batched;
extern fp_rocsolver_cgels_strided_batched g_rocsolver_cgels_strided_batched;
extern fp_rocsolver_zgels_strided_batched g_rocsolver_zgels_strided_batched;

#define rocsolver_sgetrf g_rocsolver_sgetrf
#define rocsolver_dgetrf g_rocsolver_dgetrf
#define rocsolver_cgetrf g_rocsolver_cgetrf
#define rocsolver_zgetrf g_rocsolver_zgetrf
#define rocsolver_sgetrf_batched g_rocsolver_sgetrf_batched
#define rocsolver_dgetrf_batched g_rocsolver_dgetrf_batched
#define rocsolver_cgetrf_batched g_rocsolver_cgetrf_batched
#define rocsolver_zgetrf_batched g_rocsolver_zgetrf_batched
#define rocsolver_sgetrf_strided_batched g_rocsolver_sgetrf_strided_batched
#define rocsolver_dgetrf_strided_batched g_rocsolver_dgetrf_strided_batched
#define rocsolver_cgetrf_strided_batched g_rocsolver_cgetrf_strided_batched
#define rocsolver_zgetrf_strided_batched g_rocsolver_zgetrf_strided_batched

#define rocsolver_sgetrf_npvt g_rocsolver_sgetrf_npvt
#define rocsolver_dgetrf_npvt g_rocsolver_dgetrf_npvt
#define rocsolver_cgetrf_npvt g_rocsolver_cgetrf_npvt
#define rocsolver_zgetrf_npvt g_rocsolver_zgetrf_npvt
#define rocsolver_sgetrf_npvt_batched g_rocsolver_sgetrf_npvt_batched
#define rocsolver_dgetrf_npvt_batched g_rocsolver_dgetrf_npvt_batched
#define rocsolver_cgetrf_npvt_batched g_rocsolver_cgetrf_npvt_batched
#define rocsolver_zgetrf_npvt_batched g_rocsolver_zgetrf_npvt_batched
#define rocsolver_sgetrf_npvt_strided_batched g_rocsolver_sgetrf_npvt_strided_batched
#define rocsolver_dgetrf_npvt_strided_batched g_rocsolver_dgetrf_npvt_strided_batched
#define rocsolver_cgetrf_npvt_strided_batched g_rocsolver_cgetrf_npvt_strided_batched
#define rocsolver_zgetrf_npvt_strided_batched g_rocsolver_zgetrf_npvt_strided_batched

#define rocsolver_sgetrs g_rocsolver_sgetrs
#define rocsolver_dgetrs g_rocsolver_dgetrs
#define rocsolver_cgetrs g_rocsolver_cgetrs
#define rocsolver_zgetrs g_rocsolver_zgetrs
#define rocsolver_sgetrs_batched g_rocsolver_sgetrs_batched
#define rocsolver_dgetrs_batched g_rocsolver_dgetrs_batched
#define rocsolver_cgetrs_batched g_rocsolver_cgetrs_batched
#define rocsolver_zgetrs_batched g_rocsolver_zgetrs_batched
#define rocsolver_sgetrs_strided_batched g_rocsolver_sgetrs_strided_batched
#define rocsolver_dgetrs_strided_batched g_rocsolver_dgetrs_strided_batched
#define rocsolver_cgetrs_strided_batched g_rocsolver_cgetrs_strided_batched
#define rocsolver_zgetrs_strided_batched g_rocsolver_zgetrs_strided_batched

#define rocsolver_sgetri_outofplace_batched g_rocsolver_sgetri_outofplace_batched
#define rocsolver_dgetri_outofplace_batched g_rocsolver_dgetri_outofplace_batched
#define rocsolver_cgetri_outofplace_batched g_rocsolver_cgetri_outofplace_batched
#define rocsolver_zgetri_outofplace_batched g_rocsolver_zgetri_outofplace_batched
#define rocsolver_sgetri_npvt_outofplace_batched g_rocsolver_sgetri_npvt_outofplace_batched
#define rocsolver_dgetri_npvt_outofplace_batched g_rocsolver_dgetri_npvt_outofplace_batched
#define rocsolver_cgetri_npvt_outofplace_batched g_rocsolver_cgetri_npvt_outofplace_batched
#define rocsolver_zgetri_npvt_outofplace_batched g_rocsolver_zgetri_npvt_outofplace_batched

#define rocsolver_sgeqrf g_rocsolver_sgeqrf
#define rocsolver_dgeqrf g_rocsolver_dgeqrf
#define rocsolver_cgeqrf g_rocsolver_cgeqrf
#define rocsolver_zgeqrf g_rocsolver_zgeqrf
#define rocsolver_sgeqrf_ptr_batched g_rocsolver_sgeqrf_ptr_batched
#define rocsolver_dgeqrf_ptr_batched g_rocsolver_dgeqrf_ptr_batched
#define rocsolver_cgeqrf_ptr_batched g_rocsolver_cgeqrf_ptr_batched
#define rocsolver_zgeqrf_ptr_batched g_rocsolver_zgeqrf_ptr_batched
#define rocsolver_sgeqrf_strided_batched g_rocsolver_sgeqrf_strided_batched
#define rocsolver_dgeqrf_strided_batched g_rocsolver_dgeqrf_strided_batched
#define rocsolver_cgeqrf_strided_batched g_rocsolver_cgeqrf_strided_batched
#define rocsolver_zgeqrf_strided_batched g_rocsolver_zgeqrf_strided_batched

#define rocsolver_sgels g_rocsolver_sgels
#define rocsolver_dgels g_rocsolver_dgels
#define rocsolver_cgels g_rocsolver_cgels
#define rocsolver_zgels g_rocsolver_zgels
#define rocsolver_sgels_batched g_rocsolver_sgels_batched
#define rocsolver_dgels_batched g_rocsolver_dgels_batched
#define rocsolver_cgels_batched g_rocsolver_cgels_batched
#define rocsolver_zgels_batched g_rocsolver_zgels_batched
#define rocsolver_sgels_strided_batched g_rocsolver_sgels_strided_batched
#define rocsolver_dgels_strided_batched g_rocsolver_dgels_strided_batched
#define rocsolver_cgels_strided_batched g_rocsolver_cgels_strided_batched
#define rocsolver_zgels_strided_batched g_rocsolver_zgels_strided_batched

#endif

bool try_load_rocsolver();
