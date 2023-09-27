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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasTrsvStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_transA,
                                                     e_diag,
                                                     e_M,
                                                     e_lda,
                                                     e_incx,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_trsv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrsvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsvStridedBatchedFn
        = FORTRAN ? hipblasTrsvStridedBatched<T, true> : hipblasTrsvStridedBatched<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    int                M            = arg.M;
    int                incx         = arg.incx;
    int                lda          = arg.lda;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    int           abs_incx = incx < 0 ? -incx : incx;
    hipblasStride strideA  = lda * M * stride_scale;
    hipblasStride stridex  = abs_incx * M * stride_scale;
    size_t        size_A   = size_t(strideA) * batch_count;
    size_t        size_x   = size_t(stridex) * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || lda < M || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        hipblasStatus_t actual = hipblasTrsvStridedBatchedFn(handle,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             nullptr,
                                                             lda,
                                                             strideA,
                                                             nullptr,
                                                             incx,
                                                             stridex,
                                                             batch_count);
        EXPECT_HIPBLAS_STATUS2(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);

    device_vector<T> dA(size_A);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, hipblas_error, cumulative_hipblas_error;

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, M, M, lda, strideA, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, arg, M, abs_incx, stridex, batch_count, hipblas_client_never_set_nan, false, true);
    hb = hx;

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb  = hA.data() + b * strideA;
        T* AATb = AAT.data() + b * strideA;
        T* hbb  = hb.data() + b * stridex;
        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(
            HIPBLAS_OP_N, HIPBLAS_OP_T, M, M, M, (T)1.0, hAb, lda, hAb, lda, (T)0.0, AATb, lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < M; i++)
        {
            T t = 0.0;
            for(int j = 0; j < M; j++)
            {
                hAb[i + j * lda] = AATb[i + j * lda];
                t += std::abs(AATb[i + j * lda]);
            }
            hAb[i + i * lda] = t;
        }
        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(arg.uplo, M, hAb, lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(arg.diag == 'U' || arg.diag == 'u')
        {
            if('L' == arg.uplo || 'l' == arg.uplo)
                for(int i = 0; i < M; i++)
                {
                    T diag = hAb[i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hAb[i + j * lda] = hAb[i + j * lda] / diag;
                }
            else
                for(int j = 0; j < M; j++)
                {
                    T diag = hAb[j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hAb[i + j * lda] = hAb[i + j * lda] / diag;
                }
        }

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, M, hAb, lda, hbb, incx);
    }

    hx_or_b_1 = hb;

    // copy data from CPU to device
    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(
        hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasTrsvStridedBatchedFn(
            handle, uplo, transA, diag, M, dA, lda, strideA, dx_or_b, incx, stridex, batch_count));

        // copy output from device to CPU
        ASSERT_HIP_SUCCESS(
            hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = std::abs(vector_norm_1<T>(
                M, abs_incx, hx.data() + b * stridex, hx_or_b_1.data() + b * stridex));
            if(arg.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * M;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(hipblasTrsvStridedBatchedFn(handle,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               M,
                                                               dA,
                                                               lda,
                                                               strideA,
                                                               dx_or_b,
                                                               incx,
                                                               stridex,
                                                               batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTrsvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     trsv_gflop_count<T>(M),
                                                     trsv_gbyte_count<T>(M),
                                                     cumulative_hipblas_error);
    }
}

template <typename T>
hipblasStatus_t testing_trsv_strided_batched_ret(const Arguments& arg)
{
    testing_trsv_strided_batched<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
