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

#include "arg_check.h"
#include "testing_common.hpp"
#include <typeinfo>

/* ============================================================================================ */

using hipblasGemmBatchedModel = ArgumentModel<e_a_type,
                                              e_transA,
                                              e_transB,
                                              e_M,
                                              e_N,
                                              e_K,
                                              e_alpha,
                                              e_lda,
                                              e_ldb,
                                              e_beta,
                                              e_ldc,
                                              e_batch_count>;

inline void testname_gemm_batched(const Arguments& arg, std::string& name)
{
    hipblasGemmBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_gemm_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasGemmBatchedFn
        = FORTRAN ? hipblasGemmBatched<T, true> : hipblasGemmBatched<T, false>;

    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB      = char2hipblas_operation(arg.transB);
    int                M           = arg.M;
    int                N           = arg.N;
    int                K           = arg.K;
    int                lda         = arg.lda;
    int                ldb         = arg.ldb;
    int                ldc         = arg.ldc;
    int                batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // bad arg checks
    if(batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0)
    {
        hipblasStatus_t    status = HIPBLAS_STATUS_SUCCESS;
        hipblasLocalHandle handle(arg);

        const T *dA_array[1], *dB_array[1];
        T*       dC1_array[1];

        status = hipblasGemmBatchedFn(handle,
                                      transA,
                                      transB,
                                      M,
                                      N,
                                      K,
                                      &h_alpha,
                                      dA_array,
                                      lda,
                                      dB_array,
                                      ldb,
                                      &h_beta,
                                      dC1_array,
                                      ldc,
                                      batch_count);

        verify_hipblas_status_invalid_value(
            status,
            "ERROR: batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 ");

        return;
    }

    int A_row, A_col, B_row, B_col;

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M;
        A_col = K;
    }
    else
    {
        A_row = K;
        A_col = M;
    }

    if(transB == HIPBLAS_OP_N)
    {
        B_row = K;
        B_col = N;
    }
    else
    {
        B_row = N;
        B_col = K;
    }

    if(lda < A_row || ldb < B_row || ldc < M)
    {
        return;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    size_t A_size = size_t(lda) * A_col;
    size_t B_size = size_t(ldb) * B_col;
    size_t C_size = size_t(ldc) * N;

    // host arrays
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_copy(C_size, 1, batch_count);

    // device arrays
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    ASSERT_HIP_SUCCESS(dA.memcheck());
    ASSERT_HIP_SUCCESS(dB.memcheck());
    ASSERT_HIP_SUCCESS(dC.memcheck());

    hipblas_init_vector(hA, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hB, arg, hipblas_client_alpha_sets_nan);
    hipblas_init_vector(hC_host, arg, hipblas_client_beta_sets_nan);

    hC_device.copy_from(hC_host);
    hC_copy.copy_from(hC_host);

    ASSERT_HIP_SUCCESS(dA.transfer_from(hA));
    ASSERT_HIP_SUCCESS(dB.transfer_from(hB));
    ASSERT_HIP_SUCCESS(dC.transfer_from(hC_host));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // calculate "golden" result on CPU
        for(int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T>(transA,
                          transB,
                          M,
                          N,
                          K,
                          h_alpha,
                          (T*)hA[i],
                          lda,
                          (T*)hB[i],
                          ldb,
                          h_beta,
                          (T*)hC_copy[i],
                          ldc);
        }

        // test hipBLAS batched gemm with alpha and beta pointers on device
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasGemmBatchedFn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    d_alpha,
                                                    (const T* const*)dA.ptr_on_device(),
                                                    lda,
                                                    (const T* const*)dB.ptr_on_device(),
                                                    ldb,
                                                    d_beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        ASSERT_HIP_SUCCESS(hC_device.transfer_from(dC));

        // test hipBLAS batched gemm with alpha and beta pointers on host
        ASSERT_HIP_SUCCESS(dC.transfer_from(hC_host));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasGemmBatchedFn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    &h_alpha,
                                                    (const T* const*)dA.ptr_on_device(),
                                                    lda,
                                                    (const T* const*)dB.ptr_on_device(),
                                                    ldb,
                                                    &h_beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        ASSERT_HIP_SUCCESS(hC_host.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, hC_copy, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, hC_copy, hC_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));

        // gemm has better performance in host mode. In rocBLAS in device mode
        // we need to copy alpha and beta to the host.
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(hipblasGemmBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        &h_alpha,
                                                        (const T* const*)dA.ptr_on_device(),
                                                        lda,
                                                        (const T* const*)dB.ptr_on_device(),
                                                        ldb,
                                                        &h_beta,
                                                        dC.ptr_on_device(),
                                                        ldc,
                                                        batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              gemm_gflop_count<T>(M, N, K),
                                              gemm_gbyte_count<T>(M, N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}

template <typename T>
hipblasStatus_t testing_gemm_batched_ret(const Arguments& arg)
{
    testing_gemm_batched<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
