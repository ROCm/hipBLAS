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

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSymmStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_side,
                                                     e_uplo,
                                                     e_M,
                                                     e_N,
                                                     e_alpha,
                                                     e_lda,
                                                     e_ldb,
                                                     e_beta,
                                                     e_ldc,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_symm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasSymmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_symm_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasSymmStridedBatchedFn    = arg.api == FORTRAN ? hipblasSymmStridedBatched<T, true>
                                                             : hipblasSymmStridedBatched<T, false>;
    auto hipblasSymmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasSymmStridedBatched_64<T, true>
                                              : hipblasSymmStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           M           = 101;
    int64_t           N           = 100;
    int64_t           lda         = 102;
    int64_t           ldb         = 103;
    int64_t           ldc         = 104;
    int64_t           batch_count = 2;
    hipblasSideMode_t side        = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_LOWER;

    size_t dim_A = (side == HIPBLAS_SIDE_LEFT ? N : M);

    hipblasStride stride_A = dim_A * lda;
    hipblasStride stride_B = N * ldb;
    hipblasStride stride_C = N * ldc;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(dim_A, dim_A, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(M, N, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_C, batch_count);

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

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasSymmStridedBatchedFn,
                    (nullptr,
                     side,
                     uplo,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSymmStridedBatchedFn,
                    (handle,
                     HIPBLAS_SIDE_BOTH,
                     uplo,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSymmStridedBatchedFn,
                    (handle,
                     (hipblasSideMode_t)HIPBLAS_OP_N,
                     uplo,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSymmStridedBatchedFn,
                    (handle,
                     side,
                     HIPBLAS_FILL_MODE_FULL,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSymmStridedBatchedFn,
                    (handle,
                     side,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSymmStridedBatchedFn,
                        (handle,
                         side,
                         uplo,
                         M,
                         N,
                         nullptr,
                         dA,
                         lda,
                         strideA,
                         dB,
                         ldb,
                         strideB,
                         beta,
                         dC,
                         ldc,
                         strideC,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSymmStridedBatchedFn,
                        (handle,
                         side,
                         uplo,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         strideA,
                         dB,
                         ldb,
                         strideB,
                         nullptr,
                         dC,
                         ldc,
                         strideC,
                         batch_count));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSymmStridedBatchedFn,
                            (handle,
                             side,
                             uplo,
                             M,
                             N,
                             alpha,
                             nullptr,
                             lda,
                             strideA,
                             dB,
                             ldb,
                             strideB,
                             beta,
                             dC,
                             ldc,
                             strideC,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSymmStridedBatchedFn,
                            (handle,
                             side,
                             uplo,
                             M,
                             N,
                             alpha,
                             dA,
                             lda,
                             strideA,
                             nullptr,
                             ldb,
                             strideB,
                             beta,
                             dC,
                             ldc,
                             strideC,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSymmStridedBatchedFn,
                            (handle,
                             side,
                             uplo,
                             M,
                             N,
                             alpha,
                             dA,
                             lda,
                             strideA,
                             dB,
                             ldb,
                             strideB,
                             beta,
                             nullptr,
                             ldc,
                             strideC,
                             batch_count));
            }

            // alpha == 0 && beta == 1, can have all nullptrs
            DAPI_CHECK(hipblasSymmStridedBatchedFn,
                       (handle,
                        side,
                        uplo,
                        M,
                        N,
                        zero,
                        nullptr,
                        lda,
                        strideA,
                        nullptr,
                        ldb,
                        strideB,
                        one,
                        nullptr,
                        ldc,
                        strideC,
                        batch_count));

            // 64-bit interface test
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSymmStridedBatchedFn,
                        (handle,
                         side,
                         uplo,
                         c_i32_overflow,
                         c_i32_overflow,
                         zero,
                         nullptr,
                         c_i32_overflow,
                         strideA,
                         nullptr,
                         c_i32_overflow,
                         strideB,
                         one,
                         nullptr,
                         c_i32_overflow,
                         strideC,
                         c_i32_overflow));
        }

        // If M == 0 || N == 0  batch_count == 0, can have nullptrs
        DAPI_CHECK(hipblasSymmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    0,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    ldb,
                    strideB,
                    nullptr,
                    nullptr,
                    ldc,
                    strideC,
                    batch_count));
        DAPI_CHECK(hipblasSymmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    M,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    ldb,
                    strideB,
                    nullptr,
                    nullptr,
                    ldc,
                    strideC,
                    batch_count));
        DAPI_CHECK(hipblasSymmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    M,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    ldb,
                    strideB,
                    nullptr,
                    nullptr,
                    ldc,
                    strideC,
                    0));
    }
}

template <typename T>
void testing_symm_strided_batched(const Arguments& arg)
{
    auto hipblasSymmStridedBatchedFn    = arg.api == FORTRAN ? hipblasSymmStridedBatched<T, true>
                                                             : hipblasSymmStridedBatched<T, false>;
    auto hipblasSymmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasSymmStridedBatched_64<T, true>
                                              : hipblasSymmStridedBatched_64<T, false>;

    hipblasSideMode_t side         = char2hipblas_side(arg.side);
    hipblasFillMode_t uplo         = char2hipblas_fill(arg.uplo);
    int64_t           M            = arg.M;
    int64_t           N            = arg.N;
    int64_t           lda          = arg.lda;
    int64_t           ldb          = arg.ldb;
    int64_t           ldc          = arg.ldc;
    double            stride_scale = arg.stride_scale;
    int64_t           batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    size_t dim_A = (side == HIPBLAS_SIDE_LEFT ? N : M);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < dim_A || ldb < M || ldc < M || batch_count < 0)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSymmStridedBatchedFn,
                    (handle,
                     side,
                     uplo,
                     M,
                     N,
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

    hipblasStride stride_A = lda * dim_A * stride_scale;
    hipblasStride stride_B = ldb * N * stride_scale;
    hipblasStride stride_C = ldc * N * stride_scale;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(dim_A, dim_A, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hC_host(M, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_device(M, N, ldc, stride_C, batch_count);
    host_strided_batch_matrix<T> hC_cpu(M, N, ldc, stride_C, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_host.memcheck());
    CHECK_HIP_ERROR(hC_device.memcheck());
    CHECK_HIP_ERROR(hC_cpu.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(dim_A, dim_A, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(M, N, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_C, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_symmetric_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_beta_sets_nan, hipblas_general_matrix);

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
        DAPI_CHECK(hipblasSymmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    M,
                    N,
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
        DAPI_CHECK(hipblasSymmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    M,
                    N,
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
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_symm<T>(side, uplo, M, N, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_cpu[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_cpu, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_cpu, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_cpu, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_cpu, hC_device, batch_count);
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

            DAPI_DISPATCH(hipblasSymmStridedBatchedFn,
                          (handle,
                           side,
                           uplo,
                           M,
                           N,
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
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasSymmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     symm_gflop_count<T>(M, N, dim_A),
                                                     symm_gbyte_count<T>(M, N, dim_A),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
