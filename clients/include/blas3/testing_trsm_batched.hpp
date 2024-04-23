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

using hipblasTrsmBatchedModel = ArgumentModel<e_a_type,
                                              e_side,
                                              e_uplo,
                                              e_transA,
                                              e_diag,
                                              e_M,
                                              e_N,
                                              e_alpha,
                                              e_lda,
                                              e_ldb,
                                              e_batch_count>;

inline void testname_trsm_batched(const Arguments& arg, std::string& name)
{
    hipblasTrsmBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsm_batched_bad_arg(const Arguments& arg)
{
    auto hipblasTrsmBatchedFn
        = arg.api == FORTRAN ? hipblasTrsmBatched<T, true> : hipblasTrsmBatched<T, false>;
    auto hipblasTrsmBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasTrsmBatched_64<T, true> : hipblasTrsmBatched_64<T, false>;

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

    device_batch_vector<T> dA(K * lda, 1, batch_count);
    device_batch_vector<T> dB(N * ldb, 1, batch_count);

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
                    hipblasTrsmBatchedFn,
                    (nullptr,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     batch_count));

        DAPI_EXPECT(
#ifdef __HIP_PLATFORM_NVCC__
            HIPBLAS_STATUS_INVALID_ENUM,
#else
            HIPBLAS_STATUS_INVALID_VALUE,
#endif
            hipblasTrsmBatchedFn,
            (handle,
             HIPBLAS_SIDE_BOTH,
             uplo,
             transA,
             diag,
             M,
             N,
             alpha,
             dA.ptr_on_device(),
             lda,
             dB.ptr_on_device(),
             ldb,
             batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasTrsmBatchedFn,
                    (handle,
                     side,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     diag,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasTrsmBatchedFn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                     M,
                     N,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     dB.ptr_on_device(),
                     ldb,
                     batch_count));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmBatchedFn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         nullptr,
                         dA.ptr_on_device(),
                         lda,
                         dB.ptr_on_device(),
                         ldb,
                         batch_count));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasTrsmBatchedFn,
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
                             dB.ptr_on_device(),
                             ldb,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasTrsmBatchedFn,
                            (handle,
                             side,
                             uplo,
                             transA,
                             diag,
                             M,
                             N,
                             alpha,
                             dA.ptr_on_device(),
                             lda,
                             nullptr,
                             ldb,
                             batch_count));
            }

            // If alpha == 0, then A can be nullptr
            DAPI_CHECK(hipblasTrsmBatchedFn,
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
                        dB.ptr_on_device(),
                        ldb,
                        batch_count));

            // 64-bit interface tests
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmBatchedFn,
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
                         nullptr,
                         c_i32_overflow,
                         c_i32_overflow));
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                             : HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasTrsmBatchedFn,
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
                         nullptr,
                         c_i32_overflow,
                         c_i32_overflow));
        }

        // If M == 0 || N == 0  batch_count == 0, can have nullptrs
        DAPI_CHECK(hipblasTrsmBatchedFn,
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
                    nullptr,
                    ldb,
                    batch_count));
        DAPI_CHECK(hipblasTrsmBatchedFn,
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
                    nullptr,
                    ldb,
                    batch_count));
        DAPI_CHECK(
            hipblasTrsmBatchedFn,
            (handle, side, uplo, transA, diag, M, N, nullptr, nullptr, lda, nullptr, ldb, 0));
    }
}

template <typename T>
void testing_trsm_batched(const Arguments& arg)
{
    auto hipblasTrsmBatchedFn
        = arg.api == FORTRAN ? hipblasTrsmBatched<T, true> : hipblasTrsmBatched<T, false>;
    auto hipblasTrsmBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasTrsmBatched_64<T, true> : hipblasTrsmBatched_64<T, false>;

    hipblasSideMode_t  side        = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    int64_t            M           = arg.M;
    int64_t            N           = arg.N;
    int64_t            lda         = arg.lda;
    int64_t            ldb         = arg.ldb;
    int64_t            batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();

    int64_t K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    size_t  A_size = lda * K;
    size_t  B_size = ldb * N;

    hipblasLocalHandle handle(arg);

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0;
    if(invalid_size || !batch_count)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasTrsmBatchedFn,
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
                     nullptr,
                     ldb,
                     batch_count));

        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB_host(B_size, 1, batch_count);
    host_batch_vector<T> hB_device(B_size, 1, batch_count);
    host_batch_vector<T> hB_gold(B_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_vector<T>       d_alpha(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial hA on CPU
    hipblas_init_vector(hB_host, arg, hipblas_client_never_set_nan);

    for(int b = 0; b < batch_count; b++)
    {
        hipblas_init_matrix_type(hipblas_diagonally_dominant_triangular_matrix,
                                 (T*)hA[b],
                                 arg,
                                 K,
                                 K,
                                 lda,
                                 0,
                                 1,
                                 hipblas_client_never_set_nan,
                                 true);

        if(diag == HIPBLAS_DIAG_UNIT)
        {
            make_unit_diagonal(uplo, (T*)hA[b], lda, K);
        }

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

    hB_gold.copy_from(hB_host);
    hB_device.copy_from(hB_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasTrsmBatchedFn,
                   (handle,
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
                    batch_count));

        CHECK_HIP_ERROR(hB_host.transfer_from(dB));
        CHECK_HIP_ERROR(dB.transfer_from(hB_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasTrsmBatchedFn,
                   (handle,
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
                    batch_count));

        CHECK_HIP_ERROR(hB_device.transfer_from(dB));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_trsm<T>(
                side, uplo, transA, diag, M, N, h_alpha, (const T*)hA[b], lda, hB_gold[b], ldb);
        }

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host = norm_check_general<T>('F', M, N, ldb, hB_gold, hB_host, batch_count);
        hipblas_error_device
            = norm_check_general<T>('F', M, N, ldb, hB_gold, hB_device, batch_count);
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

            DAPI_DISPATCH(hipblasTrsmBatchedFn,
                          (handle,
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
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrsmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              trsm_gflop_count<T>(M, N, K),
                                              trsm_gbyte_count<T>(M, N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}
