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

using hipblasSymmBatchedModel = ArgumentModel<e_a_type,
                                              e_side,
                                              e_uplo,
                                              e_M,
                                              e_N,
                                              e_alpha,
                                              e_lda,
                                              e_ldb,
                                              e_beta,
                                              e_ldc,
                                              e_batch_count>;

inline void testname_symm_batched(const Arguments& arg, std::string& name)
{
    hipblasSymmBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_symm_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == FORTRAN;
    auto hipblasSymmBatchedFn
        = FORTRAN ? hipblasSymmBatched<T, true> : hipblasSymmBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           M           = 101;
    int64_t           N           = 100;
    int64_t           lda         = 102;
    int64_t           ldb         = 103;
    int64_t           ldc         = 104;
    int64_t           batch_count = 2;
    hipblasSideMode_t side        = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_LOWER;

    int64_t colsA = side == HIPBLAS_SIDE_LEFT ? N : M;

    device_batch_vector<T> dA(colsA * lda, 1, batch_count);
    device_batch_vector<T> dB(N * ldb, 1, batch_count);
    device_batch_vector<T> dC(N * ldc, 1, batch_count);

    device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    const T          h_alpha(1), h_beta(2), h_one(1), h_zero(0);

    const T* alpha = &h_alpha;
    const T* beta  = &h_beta;
    const T* one   = &h_one;
    const T* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_one, one, sizeof(*one), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            one   = d_one;
            zero  = d_zero;
        }

        EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(nullptr,
                                                   side,
                                                   uplo,
                                                   M,
                                                   N,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                   HIPBLAS_SIDE_BOTH,
                                                   uplo,
                                                   M,
                                                   N,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                   (hipblasSideMode_t)HIPBLAS_OP_N,
                                                   uplo,
                                                   M,
                                                   N,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                   side,
                                                   HIPBLAS_FILL_MODE_FULL,
                                                   M,
                                                   N,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                   side,
                                                   (hipblasFillMode_t)HIPBLAS_OP_N,
                                                   M,
                                                   N,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dB.ptr_on_device(),
                                                   ldb,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                       side,
                                                       uplo,
                                                       M,
                                                       N,
                                                       nullptr,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       ldb,
                                                       beta,
                                                       dC.ptr_on_device(),
                                                       ldc,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                       side,
                                                       uplo,
                                                       M,
                                                       N,
                                                       alpha,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       ldb,
                                                       nullptr,
                                                       dC.ptr_on_device(),
                                                       ldc,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                           side,
                                                           uplo,
                                                           M,
                                                           N,
                                                           alpha,
                                                           nullptr,
                                                           lda,
                                                           dB.ptr_on_device(),
                                                           ldb,
                                                           beta,
                                                           dC.ptr_on_device(),
                                                           ldc,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                           side,
                                                           uplo,
                                                           M,
                                                           N,
                                                           alpha,
                                                           dA.ptr_on_device(),
                                                           lda,
                                                           nullptr,
                                                           ldb,
                                                           beta,
                                                           dC.ptr_on_device(),
                                                           ldc,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasSymmBatchedFn(handle,
                                                           side,
                                                           uplo,
                                                           M,
                                                           N,
                                                           alpha,
                                                           dA.ptr_on_device(),
                                                           lda,
                                                           dB.ptr_on_device(),
                                                           ldb,
                                                           beta,
                                                           nullptr,
                                                           ldc,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // alpha == 0 && beta == 1, can have all nullptrs
            CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                     side,
                                                     uplo,
                                                     M,
                                                     N,
                                                     zero,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     ldb,
                                                     one,
                                                     nullptr,
                                                     ldc,
                                                     batch_count));
        }

        // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                 side,
                                                 uplo,
                                                 0,
                                                 N,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 nullptr,
                                                 nullptr,
                                                 ldc,
                                                 batch_count));
        CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                 side,
                                                 uplo,
                                                 M,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 nullptr,
                                                 nullptr,
                                                 ldc,
                                                 batch_count));
        CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                 side,
                                                 uplo,
                                                 M,
                                                 N,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 nullptr,
                                                 nullptr,
                                                 ldc,
                                                 0));
    }
}

template <typename T>
void testing_symm_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == FORTRAN;
    auto hipblasSymmBatchedFn
        = FORTRAN ? hipblasSymmBatched<T, true> : hipblasSymmBatched<T, false>;

    hipblasSideMode_t side        = char2hipblas_side(arg.side);
    hipblasFillMode_t uplo        = char2hipblas_fill(arg.uplo);
    int               M           = arg.M;
    int               N           = arg.N;
    int               lda         = arg.lda;
    int               ldb         = arg.ldb;
    int               ldc         = arg.ldc;
    int               batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    int K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || ldc < M || batch_count < 0)
    {
        return;
    }
    else if(batch_count == 0)
    {
        return;
    }

    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;
    size_t C_size = size_t(ldc) * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // host arrays
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_gold(C_size, 1, batch_count);

    // device arrays
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    hipblas_init_vector(hA, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hB, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hC_host, arg, hipblas_client_beta_sets_nan);
    hC_device.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                 side,
                                                 uplo,
                                                 M,
                                                 N,
                                                 &h_alpha,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 dB.ptr_on_device(),
                                                 ldb,
                                                 &h_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                 side,
                                                 uplo,
                                                 M,
                                                 N,
                                                 d_alpha,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 dB.ptr_on_device(),
                                                 ldb,
                                                 d_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_symm<T>(
                side, uplo, M, N, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_gold[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasSymmBatchedFn(handle,
                                                     side,
                                                     uplo,
                                                     M,
                                                     N,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dB.ptr_on_device(),
                                                     ldb,
                                                     d_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasSymmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              symm_gflop_count<T>(M, N, K),
                                              symm_gbyte_count<T>(M, N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}
