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

using hipblasSyrkxModel = ArgumentModel<e_a_type, e_uplo, e_transA, e_N, e_K, e_lda, e_ldb, e_ldc>;

inline void testname_syrkx(const Arguments& arg, std::string& name)
{
    hipblasSyrkxModel{}.test_name(arg, name);
}

template <typename T>
void testing_syrkx_bad_arg(const Arguments& arg)
{
    auto hipblasSyrkxFn = arg.api == FORTRAN ? hipblasSyrkx<T, true> : hipblasSyrkx<T, false>;
    auto hipblasSyrkxFn_64
        = arg.api == FORTRAN_64 ? hipblasSyrkx_64<T, true> : hipblasSyrkx_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N      = 101;
    int64_t            K      = 100;
    int64_t            lda    = 102;
    int64_t            ldb    = 103;
    int64_t            ldc    = 104;
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_LOWER;

    int64_t cols = transA == HIPBLAS_OP_N ? K : N;

    device_vector<T> dA(cols * lda);
    device_vector<T> dB(cols * ldb);
    device_vector<T> dC(N * ldc);

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
                    hipblasSyrkxFn,
                    (nullptr, uplo, transA, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasSyrkxFn,
            (handle, HIPBLAS_FILL_MODE_FULL, transA, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyrkxFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyrkxFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkxFn,
                        (handle, uplo, transA, N, K, nullptr, dA, lda, dB, ldb, beta, dC, ldc));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkxFn,
                        (handle, uplo, transA, N, K, alpha, dA, lda, dB, ldb, nullptr, dC, ldc));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyrkxFn,
                    (handle, uplo, transA, N, K, alpha, nullptr, lda, dB, ldb, beta, dC, ldc));
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyrkxFn,
                    (handle, uplo, transA, N, K, alpha, dA, lda, nullptr, ldb, beta, dC, ldc));
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyrkxFn,
                    (handle, uplo, transA, N, K, alpha, dA, lda, dB, ldb, beta, nullptr, ldc));
            }

            // If k == 0 && beta == 1, A, B, C may be nullptr
            DAPI_CHECK(
                hipblasSyrkxFn,
                (handle, uplo, transA, N, 0, alpha, nullptr, lda, nullptr, ldb, one, nullptr, ldc));

            // If alpha == 0 && beta == 1, A, B, C may be nullptr
            DAPI_CHECK(
                hipblasSyrkxFn,
                (handle, uplo, transA, N, K, zero, nullptr, lda, nullptr, ldb, one, nullptr, ldc));
        }

        // If N == 0, can have nullptrs
        DAPI_CHECK(hipblasSyrkxFn,
                   (handle,
                    uplo,
                    transA,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    ldb,
                    nullptr,
                    nullptr,
                    ldc));
    }
}

template <typename T>
void testing_syrkx(const Arguments& arg)
{
    auto hipblasSyrkxFn = arg.api == FORTRAN ? hipblasSyrkx<T, true> : hipblasSyrkx<T, false>;
    auto hipblasSyrkxFn_64
        = arg.api == FORTRAN_64 ? hipblasSyrkx_64<T, true> : hipblasSyrkx_64<T, false>;

    hipblasFillMode_t  uplo  = char2hipblas_fill(arg.uplo);
    hipblasOperation_t trans = char2hipblas_operation(arg.transA);
    int64_t            N     = arg.N;
    int64_t            K     = arg.K;
    int64_t            lda   = arg.lda;
    int64_t            ldb   = arg.ldb;
    int64_t            ldc   = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size
        = (N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && (lda < N || ldb < N))
           || (transA != HIPBLAS_OP_N && (lda < K || ldb < K)));
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSyrkxFn,
                    (handle,
                     uplo transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     nullptr,
                     ldc));
        return;
    }

    int64_t K1     = (trans == HIPBLAS_OP_N ? K : N);
    size_t  A_size = size_t(lda) * K1;
    size_t  B_size = size_t(ldb) * K1;
    size_t  C_size = size_t(ldc) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, N, K1, lda);
    hipblas_init<T>(hB, N, K1, ldb);
    hipblas_init<T>(hC_host, N, N, ldc);
    hC_device = hC_gold = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(
            hipblasSyrkxFn,
            (handle, uplo, trans, N, K, (T*)&h_alpha, dA, lda, dB, ldb, (T*)&h_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasSyrkxFn,
                   (handle, uplo, trans, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        syrkx_reference<T>(uplo, trans, N, K, h_alpha, hA, lda, hB, ldb, h_beta, hC_gold, ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, ldc, hC_gold, hC_host);
            unit_check_general<T>(N, N, ldc, hC_gold, hC_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = hipblas_abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host));
            hipblas_error_device
                = hipblas_abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device));
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

            DAPI_DISPATCH(hipblasSyrkxFn,
                          (handle, uplo, trans, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSyrkxModel{}.log_args<T>(std::cout,
                                        arg,
                                        gpu_time_used,
                                        syrkx_gflop_count<T>(N, K),
                                        syrkx_gbyte_count<T>(N, K),
                                        hipblas_error_host,
                                        hipblas_error_device);
    }
}
