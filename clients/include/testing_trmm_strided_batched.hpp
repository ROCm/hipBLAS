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
hipblasStatus_t testing_trmm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmmStridedBatchedFn
        = FORTRAN ? hipblasTrmmStridedBatched<T, true> : hipblasTrmmStridedBatched<T, false>;

    bool inplace = argus.inplace;

    int    M            = argus.M;
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    int    ldc          = argus.ldc;
    int    ldOut        = inplace ? ldb : ldc;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasSideMode_t  side   = char2hipblas_side(argus.side_option);
    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int           K          = (side == HIPBLAS_SIDE_LEFT ? M : N);
    hipblasStride stride_A   = size_t(lda) * K * stride_scale;
    hipblasStride stride_B   = size_t(ldb) * N * stride_scale;
    hipblasStride stride_C   = inplace ? 1 : size_t(ldc) * N * stride_scale;
    hipblasStride stride_out = inplace ? stride_B : stride_C;

    size_t A_size   = stride_A * batch_count;
    size_t B_size   = stride_B * batch_count;
    size_t C_size   = stride_C * batch_count;
    size_t out_size = stride_out * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || ldOut < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC(C_size);

    host_vector<T> hOut_host(out_size);
    host_vector<T> hOut_device(out_size);
    host_vector<T> hOut_gold(out_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);

    device_vector<T>* dOut = inplace ? &dB : &dC;

    T h_alpha = argus.get_alpha<T>();

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, argus, K, K, lda, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, argus, M, N, ldb, stride_B, batch_count, hipblas_client_alpha_sets_nan, false, true);
    if(!inplace)
        hipblas_init_matrix(hC,
                            argus,
                            M,
                            N,
                            ldc,
                            stride_C,
                            batch_count,
                            hipblas_client_alpha_sets_nan,
                            false,
                            true);

    hOut_host   = inplace ? hB : hC;
    hOut_device = hOut_host;
    hOut_gold   = hOut_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        &h_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        *dOut,
                                                        ldOut,
                                                        stride_out,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hOut_host, *dOut, sizeof(T) * out_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * C_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        d_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        *dOut,
                                                        ldOut,
                                                        stride_out,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hOut_device, *dOut, sizeof(T) * out_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          h_alpha,
                          hA.data() + b * stride_A,
                          lda,
                          hB.data() + b * stride_B,
                          ldb);
        }

        copy_matrix_with_different_leading_dimensions(
            hB, hOut_gold, M, N, ldb, ldOut, stride_B, stride_out, batch_count);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldOut, stride_out, hOut_gold, hOut_host);
            unit_check_general<T>(M, N, batch_count, ldOut, stride_out, hOut_gold, hOut_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', M, N, ldOut, stride_out, hOut_gold, hOut_host, batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', M, N, ldOut, stride_out, hOut_gold, hOut_device, batch_count);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            *dOut,
                                                            ldOut,
                                                            stride_out,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_N,
                      e_lda,
                      e_ldb,
                      e_stride_b,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, K),
                         trmm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
