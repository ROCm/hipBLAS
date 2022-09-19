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

#define TRSM_BLOCK 128

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trsm_batched_ex(const Arguments& argus)
{
    bool FORTRAN                = argus.fortran;
    auto hipblasTrsmBatchedExFn = FORTRAN ? hipblasTrsmBatchedExFortran : hipblasTrsmBatchedEx;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char char_side   = argus.side;
    char char_uplo   = argus.uplo;
    char char_transA = argus.transA;
    char char_diag   = argus.diag;
    T    h_alpha     = argus.get_alpha<T>();
    int  batch_count = argus.batch_count;

    hipblasSideMode_t  side   = char2hipblas_side(char_side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(char_diag);

    int    K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;

    // check here to prevent undefined memory allocation error
    // TODO: Workaround for cuda tests, not actually testing return values
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!M || !N || !lda || !ldb || !batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB_host(B_size, 1, batch_count);
    host_batch_vector<T> hB_device(B_size, 1, batch_count);
    host_batch_vector<T> hB_cpu(B_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_batch_vector<T> dinvA(TRSM_BLOCK * K, 1, batch_count);
    device_vector<T>       d_alpha(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dinvA.memcheck());

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial hA on CPU
    hipblas_init_vector(hA, argus, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hB_host, argus, hipblas_client_never_set_nan);
    for(int b = 0; b < batch_count; b++)
    {
        // pad untouched area into zero
        for(int i = K; i < lda; i++)
        {
            for(int j = 0; j < K; j++)
            {
                hA[b][i + j * lda] = 0.0;
            }
        }

        // proprocess the matrix to avoid ill-conditioned matrix
        host_vector<int> ipiv(K);
        cblas_getrf(K, K, hA[b], lda, ipiv);
        for(int i = 0; i < K; i++)
        {
            for(int j = i; j < K; j++)
            {
                hA[b][i + j * lda] = hA[b][j + i * lda];
                if(diag == HIPBLAS_DIAG_UNIT)
                {
                    if(i == j)
                        hA[b][i + j * lda] = 1.0;
                }
            }
        }

        // pad untouched area into zero
        for(int i = M; i < ldb; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hB_host[b][i + j * ldb] = 0.0;
            }
        }

        // Calculate hB = hA*hX;
        cblas_trmm<T>(side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      T(1.0) / h_alpha,
                      (const T*)hA[b],
                      lda,
                      hB_host[b],
                      ldb);
    }

    hB_device.copy_from(hB_host);
    hB_cpu.copy_from(hB_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // calculate invA
    hipblasStride stride_A    = TRSM_BLOCK * lda + TRSM_BLOCK;
    hipblasStride stride_invA = TRSM_BLOCK * TRSM_BLOCK;
    int           blocks      = K / TRSM_BLOCK;

    for(int b = 0; b < batch_count; b++)
    {
        if(blocks > 0)
        {
            CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatched<T>(handle,
                                                              uplo,
                                                              diag,
                                                              TRSM_BLOCK,
                                                              dA[b],
                                                              lda,
                                                              stride_A,
                                                              dinvA[b],
                                                              TRSM_BLOCK,
                                                              stride_invA,
                                                              blocks));
        }

        if(K % TRSM_BLOCK != 0 || blocks == 0)
        {
            CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatched<T>(handle,
                                                              uplo,
                                                              diag,
                                                              K - TRSM_BLOCK * blocks,
                                                              dA[b] + stride_A * blocks,
                                                              lda,
                                                              stride_A,
                                                              dinvA[b] + stride_invA * blocks,
                                                              TRSM_BLOCK,
                                                              stride_invA,
                                                              1));
        }
    }

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrsmBatchedExFn(handle,
                                                   side,
                                                   uplo,
                                                   transA,
                                                   diag,
                                                   M,
                                                   N,
                                                   &h_alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   batch_count,
                                                   dinvA.ptr_on_device(),
                                                   TRSM_BLOCK * K,
                                                   argus.compute_type));

        CHECK_HIP_ERROR(hB_host.transfer_from(dB));
        CHECK_HIP_ERROR(dB.transfer_from(hB_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasTrsmBatchedExFn(handle,
                                                   side,
                                                   uplo,
                                                   transA,
                                                   diag,
                                                   M,
                                                   N,
                                                   d_alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   batch_count,
                                                   dinvA.ptr_on_device(),
                                                   TRSM_BLOCK * K,
                                                   argus.compute_type));

        CHECK_HIP_ERROR(hB_device.transfer_from(dB));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trsm<T>(
                side, uplo, transA, diag, M, N, h_alpha, (const T*)hA[b], lda, hB_cpu[b], ldb);
        }

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_host, batch_count);
        hipblas_error_device
            = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_device, batch_count);
        if(argus.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
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
            {
                gpu_time_used = get_time_us_sync(stream);
            }

            CHECK_HIPBLAS_ERROR(hipblasTrsmBatchedExFn(handle,
                                                       side,
                                                       uplo,
                                                       transA,
                                                       diag,
                                                       M,
                                                       N,
                                                       d_alpha,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       ldb,
                                                       batch_count,
                                                       dinvA.ptr_on_device(),
                                                       TRSM_BLOCK * K,
                                                       argus.compute_type));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side,
                      e_uplo,
                      e_transA,
                      e_diag_option,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_ldb,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trsm_gflop_count<T>(M, N, K),
                         trsm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
