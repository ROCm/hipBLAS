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

using hipblasTrmvStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_transA,
                                                     e_diag,
                                                     e_M,
                                                     e_lda,
                                                     e_incx,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_trmv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrmvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trmv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvStridedBatchedFn
        = FORTRAN ? hipblasTrmvStridedBatched<T, true> : hipblasTrmvStridedBatched<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    int                M            = arg.M;
    int                lda          = arg.lda;
    int                incx         = arg.incx;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    int           abs_incx = incx >= 0 ? incx : -incx;
    hipblasStride stride_A = size_t(lda) * M * stride_scale;
    hipblasStride stride_x = size_t(M) * abs_incx * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t X_size = stride_x * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || lda < M || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        hipblasStatus_t actual = hipblasTrmvStridedBatchedFn(handle,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             nullptr,
                                                             lda,
                                                             stride_A,
                                                             nullptr,
                                                             incx,
                                                             stride_x,
                                                             batch_count);
        EXPECT_HIPBLAS_STATUS2(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hres(X_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, M, M, lda, stride_A, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, arg, M, abs_incx, stride_x, batch_count, hipblas_client_never_set_nan, false, true);
    hres = hx;

    // copy data from CPU to device
    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasTrmvStridedBatchedFn(
            handle, uplo, transA, diag, M, dA, lda, stride_A, dx, incx, stride_x, batch_count));

        ASSERT_HIP_SUCCESS(hipMemcpy(hres.data(), dx, sizeof(T) * X_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo,
                          transA,
                          diag,
                          M,
                          hA.data() + b * stride_A,
                          lda,
                          hx.data() + b * stride_x,
                          incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, abs_incx, stride_x, hx, hres);
        }
        if(arg.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', 1, M, abs_incx, stride_x, hx, hres, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(hipblasTrmvStridedBatchedFn(
                handle, uplo, transA, diag, M, dA, lda, stride_A, dx, incx, stride_x, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrmvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     trmv_gflop_count<T>(M),
                                                     trmv_gbyte_count<T>(M),
                                                     hipblas_error);
    }
}

template <typename T>
hipblasStatus_t testing_trmv_strided_batched_ret(const Arguments& arg)
{
    testing_trmv_strided_batched<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
