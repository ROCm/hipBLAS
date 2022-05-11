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
hipblasStatus_t testing_trmv_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmvStridedBatchedFn
        = FORTRAN ? hipblasTrmvStridedBatched<T, true> : hipblasTrmvStridedBatched<T, false>;

    int    M            = argus.M;
    int    lda          = argus.lda;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int           abs_incx = incx >= 0 ? incx : -incx;
    hipblasStride stride_A = size_t(lda) * M * stride_scale;
    hipblasStride stride_x = size_t(M) * abs_incx * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t X_size = stride_x * batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    hipblasLocalHandle handle(argus);

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
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
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
        hA, argus, M, M, lda, stride_A, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, argus, M, abs_incx, stride_x, batch_count, hipblas_client_never_set_nan, false, true);
    hres = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrmvStridedBatchedFn(
            handle, uplo, transA, diag, M, dA, lda, stride_A, dx, incx, stride_x, batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hres.data(), dx, sizeof(T) * X_size, hipMemcpyDeviceToHost));

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
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, abs_incx, stride_x, hx, hres);
        }
        if(argus.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', 1, M, abs_incx, stride_x, hx, hres, batch_count);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrmvStridedBatchedFn(
                handle, uplo, transA, diag, M, dA, lda, stride_A, dx, incx, stride_x, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmv_gflop_count<T>(M),
                         trmv_gbyte_count<T>(M),
                         hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
