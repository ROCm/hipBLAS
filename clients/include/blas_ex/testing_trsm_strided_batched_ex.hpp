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

#define TRSM_BLOCK 128

/* ============================================================================================ */

using hipblasTrsmStridedBatchedExModel = ArgumentModel<e_a_type,
                                                       e_side,
                                                       e_uplo,
                                                       e_transA,
                                                       e_diag,
                                                       e_M,
                                                       e_N,
                                                       e_alpha,
                                                       e_lda,
                                                       e_ldb,
                                                       e_stride_scale,
                                                       e_batch_count>;

inline void testname_trsm_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasTrsmStridedBatchedExModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsm_strided_batched_ex(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasTrsmStridedBatchedExFn
        = FORTRAN ? hipblasTrsmStridedBatchedEx : hipblasTrsmStridedBatchedEx;

    hipblasSideMode_t  side         = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    int                M            = arg.M;
    int                N            = arg.N;
    int                lda          = arg.lda;
    int                ldb          = arg.ldb;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();

    int K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasStride strideA     = size_t(lda) * K * stride_scale;
    hipblasStride strideB     = size_t(ldb) * N * stride_scale;
    hipblasStride stride_invA = TRSM_BLOCK * size_t(K);
    size_t        A_size      = strideA * batch_count;
    size_t        B_size      = strideB * batch_count;
    size_t        invA_size   = stride_invA * batch_count;

    // check here to prevent undefined memory allocation error
    // TODO: Workaround for cuda tests, not actually testing return values
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return;
    }
    if(!batch_count)
    {
        return;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB_host(B_size);
    host_vector<T> hB_device(B_size);
    host_vector<T> hB_cpu(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dinvA(invA_size);
    device_vector<T> d_alpha(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial hA on CPU
    hipblas_init_matrix(
        hA, arg, K, K, lda, strideA, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_matrix(
        hB_host, arg, M, N, ldb, strideB, batch_count, hipblas_client_never_set_nan);

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb = hA.data() + b * strideA;
        T* hBb = hB_host.data() + b * strideB;

        // pad ountouched area into zero
        for(int i = K; i < lda; i++)
        {
            for(int j = 0; j < K; j++)
            {
                hAb[i + j * lda] = 0.0;
            }
        }

        // proprocess the matrix to avoid ill-conditioned matrix
        host_vector<int> ipiv(K);
        cblas_getrf(K, K, hAb, lda, ipiv.data());
        for(int i = 0; i < K; i++)
        {
            for(int j = i; j < K; j++)
            {
                hAb[i + j * lda] = hAb[j + i * lda];
                if(diag == HIPBLAS_DIAG_UNIT)
                {
                    if(i == j)
                        hAb[i + j * lda] = 1.0;
                }
            }
        }

        // pad untouched area into zero
        for(int i = M; i < ldb; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hBb[i + j * ldb] = 0.0;
            }
        }

        // Calculate hB = hA*hX;
        cblas_trmm<T>(
            side, uplo, transA, diag, M, N, T(1.0) / h_alpha, (const T*)hAb, lda, hBb, ldb);
    }

    hB_device = hB_cpu = hB_host;

    // copy data from CPU to device
    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dB, hB_host, sizeof(T) * B_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // calculate invA
    int sub_stride_A    = TRSM_BLOCK * lda + TRSM_BLOCK;
    int sub_stride_invA = TRSM_BLOCK * TRSM_BLOCK;
    int blocks          = K / TRSM_BLOCK;

    for(int b = 0; b < batch_count; b++)
    {
        if(blocks > 0)
        {
            ASSERT_HIPBLAS_SUCCESS(hipblasTrtriStridedBatched<T>(handle,
                                                                 uplo,
                                                                 diag,
                                                                 TRSM_BLOCK,
                                                                 dA + b * strideA,
                                                                 lda,
                                                                 sub_stride_A,
                                                                 dinvA + b * stride_invA,
                                                                 TRSM_BLOCK,
                                                                 sub_stride_invA,
                                                                 blocks));
        }

        if(K % TRSM_BLOCK != 0 || blocks == 0)
        {
            ASSERT_HIPBLAS_SUCCESS(
                hipblasTrtriStridedBatched<T>(handle,
                                              uplo,
                                              diag,
                                              K - TRSM_BLOCK * blocks,
                                              dA + sub_stride_A * blocks + b * strideA,
                                              lda,
                                              sub_stride_A,
                                              dinvA + sub_stride_invA * blocks + b * stride_invA,
                                              TRSM_BLOCK,
                                              sub_stride_invA,
                                              1));
        }
    }

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasTrsmStridedBatchedExFn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             &h_alpha,
                                                             dA,
                                                             lda,
                                                             strideA,
                                                             dB,
                                                             ldb,
                                                             strideB,
                                                             batch_count,
                                                             dinvA,
                                                             invA_size,
                                                             stride_invA,
                                                             arg.compute_type));

        // copy output from device to CPU
        ASSERT_HIP_SUCCESS(hipMemcpy(hB_host, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));
        ASSERT_HIP_SUCCESS(hipMemcpy(dB, hB_device, sizeof(T) * B_size, hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasTrsmStridedBatchedExFn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             d_alpha,
                                                             dA,
                                                             lda,
                                                             strideA,
                                                             dB,
                                                             ldb,
                                                             strideB,
                                                             batch_count,
                                                             dinvA,
                                                             invA_size,
                                                             stride_invA,
                                                             arg.compute_type));

        ASSERT_HIP_SUCCESS(hipMemcpy(hB_device, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trsm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          h_alpha,
                          (const T*)hA.data() + b * strideA,
                          lda,
                          hB_cpu.data() + b * strideB,
                          ldb);
        }

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host
            = norm_check_general<T>('F', M, N, ldb, strideB, hB_cpu, hB_host, batch_count);
        hipblas_error_device
            = norm_check_general<T>('F', M, N, ldb, strideB, hB_cpu, hB_device, batch_count);
        if(arg.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
            {
                gpu_time_used = get_time_us_sync(stream);
            }

            ASSERT_HIPBLAS_SUCCESS(hipblasTrsmStridedBatchedExFn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 d_alpha,
                                                                 dA,
                                                                 lda,
                                                                 strideA,
                                                                 dB,
                                                                 ldb,
                                                                 strideB,
                                                                 batch_count,
                                                                 dinvA,
                                                                 invA_size,
                                                                 stride_invA,
                                                                 arg.compute_type));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrsmStridedBatchedExModel{}.log_args<T>(std::cout,
                                                       arg,
                                                       gpu_time_used,
                                                       trsm_gflop_count<T>(M, N, K),
                                                       trsm_gbyte_count<T>(M, N, K),
                                                       hipblas_error_host,
                                                       hipblas_error_device);
    }
}

template <typename T>
hipblasStatus_t testing_trsm_strided_batched_ex_ret(const Arguments& arg)
{
    testing_trsm_strided_batched_ex<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
