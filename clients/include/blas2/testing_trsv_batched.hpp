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

using hipblasTrsvBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_lda, e_incx, e_batch_count>;

inline void testname_trsv_batched(const Arguments& arg, std::string& name)
{
    hipblasTrsvBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsvBatchedFn
        = FORTRAN ? hipblasTrsvBatched<T, true> : hipblasTrsvBatched<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    int                N           = arg.N;
    int                incx        = arg.incx;
    int                lda         = arg.lda;
    int                batch_count = arg.batch_count;

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t size_A   = size_t(lda) * N;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasTrsvBatchedFn(
            handle, uplo, transA, diag, N, nullptr, lda, nullptr, incx, batch_count);
        EXPECT_HIPBLAS_STATUS2(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> AAT(size_A, 1, batch_count);
    host_batch_vector<T> hb(N, incx, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hx_or_b_1(N, incx, batch_count);

    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx_or_b(N, incx, batch_count);

    ASSERT_HIP_SUCCESS(dA.memcheck());
    ASSERT_HIP_SUCCESS(dx_or_b.memcheck());

    double gpu_time_used, hipblas_error, cumulative_hipblas_error;

    // Initial Data on CPU
    hipblas_init_vector(hA, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);
    hb.copy_from(hx);

    for(int b = 0; b < batch_count; b++)
    {
        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(HIPBLAS_OP_N,
                      HIPBLAS_OP_T,
                      N,
                      N,
                      N,
                      (T)1.0,
                      (T*)hA[b],
                      lda,
                      (T*)hA[b],
                      lda,
                      (T)0.0,
                      (T*)AAT[b],
                      lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < N; i++)
        {
            T t = 0.0;
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * lda] = AAT[b][i + j * lda];
                t += std::abs(AAT[b][i + j * lda]);
            }
            hA[b][i + i * lda] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(arg.uplo, N, hA[b], lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(arg.diag == 'U' || arg.diag == 'u')
        {
            if('L' == arg.uplo || 'l' == arg.uplo)
                for(int i = 0; i < N; i++)
                {
                    T diag = hA[b][i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
            else
                for(int j = 0; j < N; j++)
                {
                    T diag = hA[b][j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
        }
    }

    for(int b = 0; b < batch_count; b++)
    {
        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, N, hA[b], lda, hb[b], incx);
    }

    hx_or_b_1.copy_from(hb);

    ASSERT_HIP_SUCCESS(dx_or_b.transfer_from(hx_or_b_1));
    ASSERT_HIP_SUCCESS(dA.transfer_from(hA));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasTrsvBatchedFn(handle,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    N,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx_or_b.ptr_on_device(),
                                                    incx,
                                                    batch_count));

        ASSERT_HIP_SUCCESS(hx_or_b_1.transfer_from(dx_or_b));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = std::abs(vector_norm_1<T>(N, abs_incx, hx[b], hx_or_b_1[b]));
            if(arg.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
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

            ASSERT_HIPBLAS_SUCCESS(hipblasTrsvBatchedFn(handle,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        N,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dx_or_b.ptr_on_device(),
                                                        incx,
                                                        batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTrsvBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              trsv_gflop_count<T>(N),
                                              trsv_gbyte_count<T>(N),
                                              cumulative_hipblas_error);
    }
}
