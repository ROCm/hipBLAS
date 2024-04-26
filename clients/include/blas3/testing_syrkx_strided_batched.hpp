/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "syrkx_reference.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSyrkxStridedBatchedModel = ArgumentModel<e_a_type,
                                                      e_uplo,
                                                      e_transA,
                                                      e_N,
                                                      e_K,
                                                      e_lda,
                                                      e_ldb,
                                                      e_ldc,
                                                      e_stride_scale,
                                                      e_batch_count>;

inline void testname_syrkx_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasSyrkxStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_syrkx_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasSyrkxStridedBatchedFn    = arg.api == FORTRAN ? hipblasSyrkxStridedBatched<T, true>
                                                              : hipblasSyrkxStridedBatched<T, false>;
    auto hipblasSyrkxStridedBatchedFn_64 = arg.api == FORTRAN_64
                                               ? hipblasSyrkxStridedBatched_64<T, true>
                                               : hipblasSyrkxStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N           = 101;
    int64_t            K           = 100;
    int64_t            lda         = 102;
    int64_t            ldb         = 103;
    int64_t            ldc         = 104;
    int64_t            batch_count = 2;
    hipblasOperation_t transA      = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;

    size_t cols = (transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);
    size_t rows = (transA != HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);

    hipblasStride stride_A = cols * lda;
    hipblasStride stride_B = cols * ldb;
    hipblasStride stride_C = N * ldc;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(rows, cols, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(rows, cols, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dC(N, N, ldc, stride_C, batch_count);

    device_vector<T> d_alpha(1), d_zero(1), d_beta(1), d_one(1);
    const T          h_alpha(1), h_zero(0), h_beta(2), h_one(1);

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

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasSyrkxStridedBatchedFn,
                    (nullptr,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyrkxStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyrkxStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyrkxStridedBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     stride_A,
                     dB,
                     ldb,
                     stride_B,
                     beta,
                     dC,
                     ldc,
                     stride_C,
                     batch_count));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkxStridedBatchedFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         nullptr,
                         dA,
                         lda,
                         stride_A,
                         dB,
                         ldb,
                         stride_B,
                         beta,
                         dC,
                         ldc,
                         stride_C,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkxStridedBatchedFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         stride_A,
                         dB,
                         ldb,
                         stride_B,
                         nullptr,
                         dC,
                         ldc,
                         stride_C,
                         batch_count));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyrkxStridedBatchedFn,
                            (handle,
                             uplo,
                             transA,
                             N,
                             K,
                             alpha,
                             nullptr,
                             lda,
                             stride_A,
                             dB,
                             ldb,
                             stride_B,
                             beta,
                             dC,
                             ldc,
                             stride_C,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyrkxStridedBatchedFn,
                            (handle,
                             uplo,
                             transA,
                             N,
                             K,
                             alpha,
                             dA,
                             lda,
                             stride_A,
                             nullptr,
                             ldb,
                             stride_B,
                             beta,
                             dC,
                             ldc,
                             stride_C,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyrkxStridedBatchedFn,
                            (handle,
                             uplo,
                             transA,
                             N,
                             K,
                             alpha,
                             dA,
                             lda,
                             stride_A,
                             dB,
                             ldb,
                             stride_B,
                             beta,
                             nullptr,
                             ldc,
                             stride_C,
                             batch_count));
            }

            // If k == 0 && beta == 1, A, B, C may be nullptr
            DAPI_CHECK(hipblasSyrkxStridedBatchedFn,
                       (handle,
                        uplo,
                        transA,
                        N,
                        0,
                        alpha,
                        nullptr,
                        lda,
                        stride_A,
                        nullptr,
                        ldb,
                        stride_B,
                        one,
                        nullptr,
                        ldc,
                        stride_C,
                        batch_count));

            // If alpha == 0 && beta == 1, A, B, C may be nullptr
            DAPI_CHECK(hipblasSyrkxStridedBatchedFn,
                       (handle,
                        uplo,
                        transA,
                        N,
                        K,
                        zero,
                        nullptr,
                        lda,
                        stride_A,
                        nullptr,
                        ldb,
                        stride_B,
                        one,
                        nullptr,
                        ldc,
                        stride_C,
                        batch_count));

            // 64-bit interface test
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkxStridedBatchedFn,
                        (handle,
                         uplo,
                         transA,
                         c_i32_overflow,
                         c_i32_overflow,
                         zero,
                         nullptr,
                         c_i32_overflow,
                         stride_A,
                         nullptr,
                         c_i32_overflow,
                         stride_B,
                         one,
                         nullptr,
                         c_i32_overflow,
                         stride_C,
                         c_i32_overflow));
        }

        // If N == 0 || batch_count, can have nullptrs
        DAPI_CHECK(hipblasSyrkxStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    ldb,
                    stride_B,
                    nullptr,
                    nullptr,
                    ldc,
                    stride_C,
                    batch_count));
        DAPI_CHECK(hipblasSyrkxStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    ldb,
                    stride_B,
                    nullptr,
                    nullptr,
                    ldc,
                    stride_C,
                    0));
    }
}

template <typename T>
void testing_syrkx_strided_batched(const Arguments& arg)
{
    auto hipblasSyrkxStridedBatchedFn    = arg.api == FORTRAN ? hipblasSyrkxStridedBatched<T, true>
                                                              : hipblasSyrkxStridedBatched<T, false>;
    auto hipblasSyrkxStridedBatchedFn_64 = arg.api == FORTRAN_64
                                               ? hipblasSyrkxStridedBatched_64<T, true>
                                               : hipblasSyrkxStridedBatched_64<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    int64_t            N            = arg.N;
    int64_t            K            = arg.K;
    int64_t            lda          = arg.lda;
    int64_t            ldb          = arg.ldb;
    int64_t            ldc          = arg.ldc;
    double             stride_scale = arg.stride_scale;
    int64_t            batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    size_t rows = (transA != HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);
    size_t cols = (transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);

    hipblasStride stride_A = lda * cols * stride_scale;
    hipblasStride stride_B = ldb * cols * stride_scale;
    hipblasStride stride_C = ldc * N * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size
        = (N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && (lda < N || ldb < N))
           || (transA != HIPBLAS_OP_N && (lda < K || ldb < K)))
          || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSyrkxStridedBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     ldb,
                     stride_B,
                     nullptr,
                     nullptr,
                     ldc,
                     stride_C,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(rows, cols, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB(rows, cols, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hC_host(N, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_device(N, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_cpu(N, N, ldc, stride_C, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_host.memcheck());
    CHECK_HIP_ERROR(hC_device.memcheck());
    CHECK_HIP_ERROR(hC_cpu.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(rows, cols, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(rows, cols, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dC(N, N, ldc, stride_C, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(hB, arg, hipblas_client_never_set_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_never_set_nan, hipblas_symmetric_matrix);

    hC_device.copy_from(hC_host);
    hC_cpu.copy_from(hC_host);

    // copy data from CPU to device
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
        DAPI_CHECK(hipblasSyrkxStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    &h_alpha,
                    dA,
                    lda,
                    stride_A,
                    dB,
                    ldb,
                    stride_B,
                    &h_beta,
                    dC,
                    ldc,
                    stride_C,
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasSyrkxStridedBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    d_alpha,
                    dA,
                    lda,
                    stride_A,
                    dB,
                    ldb,
                    stride_B,
                    d_beta,
                    dC,
                    ldc,
                    stride_C,
                    batch_count));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t b = 0; b < batch_count; b++)
        {
            // B must == A to use syrk as reference
            syrkx_reference<T>(
                uplo, transA, N, K, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_cpu[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, stride_C, hC_cpu, hC_host);
            unit_check_general<T>(N, N, batch_count, ldc, stride_C, hC_cpu, hC_host);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = hipblas_abs(
                norm_check_general<T>('F', N, N, ldc, stride_C, hC_cpu, hC_host, batch_count));
            hipblas_error_device = hipblas_abs(
                norm_check_general<T>('F', N, N, ldc, stride_C, hC_cpu, hC_device, batch_count));
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

            DAPI_DISPATCH(hipblasSyrkxStridedBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           d_alpha,
                           dA,
                           lda,
                           stride_A,
                           dB,
                           ldb,
                           stride_B,
                           d_beta,
                           dC,
                           ldc,
                           stride_C,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSyrkxStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      syrkx_gflop_count<T>(N, K),
                                                      syrkx_gbyte_count<T>(N, K),
                                                      hipblas_error_host,
                                                      hipblas_error_device);
    }
}
