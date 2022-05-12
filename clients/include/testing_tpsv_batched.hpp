/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T>
hipblasStatus_t testing_tpsv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTpsvBatchedFn
        = FORTRAN ? hipblasTpsvBatched<T, true> : hipblasTpsvBatched<T, false>;

    int                N           = argus.N;
    int                incx        = argus.incx;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);
    int                batch_count = argus.batch_count;

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t size_A   = size_t(N) * N;
    size_t size_AP  = size_t(N) * (N + 1) / 2;
    size_t size_x   = abs_incx * size_t(N);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasTpsvBatchedFn(
            handle, uplo, transA, diag, N, nullptr, nullptr, incx, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hAP(size_AP, 1, batch_count);
    host_batch_vector<T> AAT(size_A, 1, batch_count);
    host_batch_vector<T> hb(N, incx, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hx_or_b_1(N, incx, batch_count);
    host_batch_vector<T> hx_or_b_2(N, incx, batch_count);
    host_batch_vector<T> cpu_x_or_b(N, incx, batch_count);

    device_batch_vector<T> dAP(size_AP, 1, batch_count);
    device_batch_vector<T> dx_or_b(N, incx, batch_count);

    CHECK_HIP_ERROR(dAP.memcheck());
    CHECK_HIP_ERROR(dx_or_b.memcheck());

    double gpu_time_used, hipblas_error, cumulative_hipblas_error = 0;

    // Initial Data on CPU
    hipblas_init_vector(hA, argus, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, argus, hipblas_client_never_set_nan, false, true);
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
                      N,
                      (T*)hA[b],
                      N,
                      (T)0.0,
                      (T*)AAT[b],
                      N);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < N; i++)
        {
            T t = 0.0;
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * N] = AAT[b][i + j * N];
                t += std::abs(AAT[b][i + j * N]);
            }
            hA[b][i + i * N] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, N, hA[b], N);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < N; i++)
                {
                    T diag = hA[b][i + i * N];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * N] = hA[b][i + j * N] / diag;
                }
            else
                for(int j = 0; j < N; j++)
                {
                    T diag = hA[b][j + j * N];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * N] = hA[b][i + j * N] / diag;
                }
        }

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hb[b], incx);

        regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], (T*)hAP[b], N);
    }

    cpu_x_or_b.copy_from(hb);
    hx_or_b_1.copy_from(hb);
    hx_or_b_2.copy_from(hb);

    CHECK_HIP_ERROR(dAP.transfer_from(hAP));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasTpsvBatchedFn(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 N,
                                                 dAP.ptr_on_device(),
                                                 dx_or_b.ptr_on_device(),
                                                 incx,
                                                 batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = std::abs(vector_norm_1<T>(N, abs_incx, hx[b], hx_or_b_1[b]));
            if(argus.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTpsvBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     N,
                                                     dAP.ptr_on_device(),
                                                     dx_or_b.ptr_on_device(),
                                                     incx,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option, e_transA_option, e_diag_option, e_N, e_incx, e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         tpsv_gflop_count<T>(N),
                         tpsv_gbyte_count<T>(N),
                         cumulative_hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
