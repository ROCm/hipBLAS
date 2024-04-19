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

using hipblasHemmModel
    = ArgumentModel<e_a_type, e_side, e_uplo, e_M, e_N, e_alpha, e_lda, e_ldb, e_beta, e_ldc>;

inline void testname_hemm(const Arguments& arg, std::string& name)
{
    hipblasHemmModel{}.test_name(arg, name);
}

template <typename T>
void testing_hemm_bad_arg(const Arguments& arg)
{
    auto hipblasHemmFn = arg.api == FORTRAN ? hipblasHemm<T, true> : hipblasHemm<T, false>;
    auto hipblasHemmFn_64
        = arg.api == FORTRAN_64 ? hipblasHemm_64<T, true> : hipblasHemm_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           M    = 101;
    int64_t           N    = 100;
    int64_t           lda  = 102;
    int64_t           ldb  = 103;
    int64_t           ldc  = 104;
    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;

    int64_t colsA = side == HIPBLAS_SIDE_LEFT ? N : M;

    device_vector<T> dA(colsA * lda);
    device_vector<T> dB(N * ldb);
    device_vector<T> dC(N * ldc);

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
                    hipblasHemmFn,
                    (nullptr, side, uplo, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(
#ifdef __HIP_PLATFORM_NVCC__
            HIPBLAS_STATUS_INVALID_ENUM,
#else
            HIPBLAS_STATUS_INVALID_VALUE,
#endif
            hipblasHemmFn,
            (handle, HIPBLAS_SIDE_BOTH, uplo, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHemmFn,
                    (handle,
                     (hipblasSideMode_t)HIPBLAS_OP_N,
                     uplo,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasHemmFn,
            (handle, side, HIPBLAS_FILL_MODE_FULL, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHemmFn,
                    (handle,
                     side,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     M,
                     N,
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
                        hipblasHemmFn,
                        (handle, side, uplo, M, N, nullptr, dA, lda, dB, ldb, beta, dC, ldc));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHemmFn,
                        (handle, side, uplo, M, N, alpha, dA, lda, dB, ldb, nullptr, dC, ldc));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHemmFn,
                    (handle, side, uplo, M, N, alpha, nullptr, lda, dB, ldb, beta, dC, ldc));
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHemmFn,
                    (handle, side, uplo, M, N, alpha, dA, lda, nullptr, ldb, beta, dC, ldc));
                DAPI_EXPECT(
                    HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHemmFn,
                    (handle, side, uplo, M, N, alpha, dA, lda, dB, ldb, beta, nullptr, ldc));
            }

            // alpha == 0 && beta == 1, can have all nullptrs
            DAPI_CHECK(
                hipblasHemmFn,
                (handle, side, uplo, M, N, zero, nullptr, lda, nullptr, ldb, one, nullptr, ldc));
        }

        // If M == 0 || N == 0, can have nullptrs
        DAPI_CHECK(
            hipblasHemmFn,
            (handle, side, uplo, 0, N, nullptr, nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc));
        DAPI_CHECK(
            hipblasHemmFn,
            (handle, side, uplo, M, 0, nullptr, nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc));
    }
}

template <typename T>
void testing_hemm(const Arguments& arg)
{
    auto hipblasHemmFn = arg.api == FORTRAN ? hipblasHemm<T, true> : hipblasHemm<T, false>;
    auto hipblasHemmFn_64
        = arg.api == FORTRAN_64 ? hipblasHemm_64<T, true> : hipblasHemm_64<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);
    hipblasFillMode_t uplo = char2hipblas_fill(arg.uplo);
    int64_t           M    = arg.M;
    int64_t           N    = arg.N;
    int64_t           lda  = arg.lda;
    int64_t           ldb  = arg.ldb;
    int64_t           ldc  = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    size_t  rows = (side == HIPBLAS_SIDE_LEFT ? N : M);
    int64_t K    = (side == HIPBLAS_SIDE_LEFT ? M : N);

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || ldc < M || ldb < M || lda < K;
    if(invalid_size || !M || !N)
    {
        DAPI_EXPECT(
            invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
            hipblasHemmFn,
            (handle, side, uplo, M, N, nullptr, nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;
    size_t C_size = size_t(ldc) * N;

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
    hipblas_init_matrix(hA, arg, rows, K, lda, 0, 1, hipblas_client_never_set_nan, true);
    hipblas_init_matrix(hB, arg, M, N, ldb, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_matrix(hC_host, arg, M, N, ldc, 0, 1, hipblas_client_beta_sets_nan);
    hC_gold   = hC_host;
    hC_device = hC_host;

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
        DAPI_CHECK(hipblasHemmFn,
                   (handle, side, uplo, M, N, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHemmFn,
                   (handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_hemm<T>(
            side, uplo, M, N, h_alpha, hA.data(), lda, hB.data(), ldb, h_beta, hC_gold.data(), ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_host);
            unit_check_general<T>(M, N, ldc, hC_gold, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_host);
            hipblas_error_device = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_device);
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

            DAPI_DISPATCH(hipblasHemmFn,
                          (handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasHemmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       hemm_gflop_count<T>(M, N, K),
                                       hemm_gbyte_count<T>(M, N, K),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
