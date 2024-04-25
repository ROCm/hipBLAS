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

using hipblasTrsmStridedBatchedModel = ArgumentModel<e_a_type,
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

inline void testname_trsm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrsmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsm_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasTrsmStridedBatchedFn    = arg.api == FORTRAN ? hipblasTrsmStridedBatched<T, true>
                                                             : hipblasTrsmStridedBatched<T, false>;
    auto hipblasTrsmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTrsmStridedBatched_64<T, true>
                                              : hipblasTrsmStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            M           = 101;
    int64_t            N           = 100;
    int64_t            lda         = 102;
    int64_t            ldb         = 103;
    int64_t            batch_count = 2;
    hipblasSideMode_t  side        = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;
    hipblasOperation_t transA      = HIPBLAS_OP_N;
    hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    hipblasStride stride_A = K * lda;
    hipblasStride stride_B = N * ldb;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(K, K, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(M, N, ldb, stride_B, batch_count);

    device_vector<T> d_alpha(1), d_zero(1);
    const T          h_alpha(1), h_zero(0);

    const T* alpha = &h_alpha;
    const T* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasTrsmStridedBatchedFn,
                    (nullptr,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     batch_count));

        DAPI_EXPECT(
#ifdef __HIP_PLATFORM_NVCC__
            HIPBLAS_STATUS_INVALID_ENUM,
#else
            HIPBLAS_STATUS_INVALID_VALUE,
#endif
            hipblasTrsmStridedBatchedFn,
            (handle,
             HIPBLAS_SIDE_BOTH,
             uplo,
             transA,
             diag,
             M,
             N,
             alpha,
             dA,
             lda,
             strideA,
             dB,
             ldb,
             strideB,
             batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrsmStridedBatchedFn,
                    (handle,
                     side,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     batch_count));
        EXPECT_HIPBLAS_STATUS(
            hipblasTrsmStridedBatchedFn(handle,
                                        side,
                                        uplo,
                                        (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                        diag,
                                        M,
                                        N,
                                        alpha,
                                        dA,
                                        lda,
                                        stride_A,
                                        dB,
                                        ldb,
                                        stride_B,
                                        batch_count),
            HIPBLAS_STATUS_INVALID_ENUM);
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsmStridedBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dB,
                     ldb,
                     strideB,
                     batch_count));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmStridedBatchedFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         nullptr,
                         dA,
                         lda,
                         strideA,
                         dB,
                         ldb,
                         strideB,
                         batch_count));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasTrsmStridedBatchedFn,
                            (handle,
                             side,
                             uplo,
                             transA,
                             diag,
                             M,
                             N,
                             alpha,
                             nullptr,
                             lda,
                             strideA,
                             dB,
                             ldb,
                             strideB,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasTrsmStridedBatchedFn,
                            (handle,
                             side,
                             uplo,
                             transA,
                             diag,
                             M,
                             N,
                             alpha,
                             dA,
                             lda,
                             strideA,
                             nullptr,
                             ldb,
                             strideB,
                             batch_count));
            }

            // If alpha == 0, then A can be nullptr
            DAPI_CHECK(hipblasTrsmStridedBatchedFn,
                       (handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        M,
                        N,
                        zero,
                        nullptr,
                        lda,
                        strideA,
                        dB,
                        ldb,
                        strideB,
                        batch_count));

            // 64-bit interface tests
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmStridedBatchedFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         0,
                         c_i32_overflow,
                         nullptr,
                         nullptr,
                         c_i32_overflow,
                         strideA,
                         nullptr,
                         c_i32_overflow,
                         strideB,
                         c_i32_overflow));
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmStridedBatchedFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         c_i32_overflow,
                         0,
                         nullptr,
                         nullptr,
                         c_i32_overflow,
                         strideA,
                         nullptr,
                         c_i32_overflow,
                         strideB,
                         c_i32_overflow));
        }

        // If M == 0 || N == 0  batch_count == 0, can have nullptrs
        DAPI_CHECK(hipblasTrsmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    0,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    ldb,
                    strideB,
                    batch_count));
        DAPI_CHECK(hipblasTrsmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    ldb,
                    strideB,
                    batch_count));
        DAPI_CHECK(hipblasTrsmStridedBatchedFn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    ldb,
                    strideB,
                    0));
    }
}

template <typename T>
void testing_trsm_strided_batched(const Arguments& arg)
{
    auto hipblasTrsmStridedBatchedFn    = arg.api == FORTRAN ? hipblasTrsmStridedBatched<T, true>
                                                             : hipblasTrsmStridedBatched<T, false>;
    auto hipblasTrsmStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasTrsmStridedBatched_64<T, true>
                                              : hipblasTrsmStridedBatched_64<T, false>;

    hipblasSideMode_t  side         = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    int64_t            M            = arg.M;
    int64_t            N            = arg.N;
    int64_t            lda          = arg.lda;
    int64_t            ldb          = arg.ldb;
    double             stride_scale = arg.stride_scale;
    int64_t            batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();

    int64_t K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasStride strideA = lda * K * stride_scale;
    hipblasStride strideB = ldb * N * stride_scale;

    hipblasLocalHandle handle(arg);

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0;
    if(invalid_size || !batch_count)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTrsmStridedBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     ldb,
                     strideB,
                     batch_count));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(K, K, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB_host(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hB_device(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hB_cpu(M, N, ldb, stride_B, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB_host.memcheck());
    CHECK_HIP_ERROR(hB_device.memcheck());
    CHECK_HIP_ERROR(hB_cpu.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(K, K, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dB(M, N, ldb, stride_B, batch_count);
    device_vector<T>               d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_diagonally_dominant_triangular_matrix, true);
    hipblas_init_matrix(
        hB_host, arg, hipblas_client_never_set_nan, hipblas_general_matrix, false, true);

    for(int64_t b = 0; b < batch_count; b++)
    {
        make_unit_diagonal(uplo, hA);
    }

    for(int b = 0; b < batch_count; b++)
    {
        // Calculate hB = hA*hX;
        ref_trmm<T>(side,
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

    hB_cpu.copy_from(hB_host);
    hB_device.copy_from(hB_host);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasTrsmStridedBatchedFn,
                   (handle,
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
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB_host.transfer_from(dB));
        CHECK_HIP_ERROR(dB.transfer_from(hB_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasTrsmStridedBatchedFn,
                   (handle,
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
                    batch_count));

        CHECK_HIP_ERROR(hB_device.transfer_from(dB));
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_trsm<T>(
                side, uplo, transA, diag, M, N, h_alpha, (const T*)hA[b], lda, hB_cpu[b], ldb);
        }

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host
            = norm_check_general<T>('F', M, N, ldb, stride_B, hB_cpu, hB_host, batch_count);
        hipblas_error_device
            = norm_check_general<T>('F', M, N, ldb, stride_B, hB_cpu, hB_device, batch_count);
        if(arg.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
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
            {
                gpu_time_used = get_time_us_sync(stream);
            }

            DAPI_DISPATCH(hipblasTrsmStridedBatchedFn,
                          (handle,
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
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrsmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     trsm_gflop_count<T>(M, N, K),
                                                     trsm_gbyte_count<T>(M, N, K),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
