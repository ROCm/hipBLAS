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

#include "syrkx_reference.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSyrkxModel = ArgumentModel<e_a_type, e_uplo, e_transA, e_N, e_K, e_lda, e_ldb, e_ldc>;

inline void testname_syrkx(const Arguments& arg, std::string& name)
{
    hipblasSyrkxModel{}.test_name(arg, name);
}

template <typename T>
void testing_syrkx(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasSyrkxFn = FORTRAN ? hipblasSyrkx<T, true> : hipblasSyrkx<T, false>;

    hipblasFillMode_t  uplo  = char2hipblas_fill(arg.uplo);
    hipblasOperation_t trans = char2hipblas_operation(arg.transA);
    int                N     = arg.N;
    int                K     = arg.K;
    int                lda   = arg.lda;
    int                ldb   = arg.ldb;
    int                ldc   = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (trans == HIPBLAS_OP_N && (lda < N || ldb < N))
       || (trans != HIPBLAS_OP_N && (lda < K || ldb < K)))
    {
        return;
    }

    int    K1     = (trans == HIPBLAS_OP_N ? K : N);
    size_t A_size = size_t(lda) * K1;
    size_t B_size = size_t(ldb) * K1;
    size_t C_size = size_t(ldc) * N;

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
    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasSyrkxFn(
            handle, uplo, trans, N, K, (T*)&h_alpha, dA, lda, dB, ldb, (T*)&h_beta, dC, ldc));

        // copy output from device to CPU
        ASSERT_HIP_SUCCESS(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
        ASSERT_HIP_SUCCESS(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(
            hipblasSyrkxFn(handle, uplo, trans, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        ASSERT_HIP_SUCCESS(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

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
            hipblas_error_host = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host));
            hipblas_error_device
                = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device));
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
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(hipblasSyrkxFn(
                handle, uplo, trans, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
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
