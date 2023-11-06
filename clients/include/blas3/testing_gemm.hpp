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
#include <typeinfo>

/* ============================================================================================ */

using hipblasGemmModel = ArgumentModel<e_a_type,
                                       e_transA,
                                       e_transB,
                                       e_M,
                                       e_N,
                                       e_K,
                                       e_alpha,
                                       e_lda,
                                       e_ldb,
                                       e_beta,
                                       e_ldc>;

inline void testname_gemm(const Arguments& arg, std::string& name)
{
    hipblasGemmModel{}.test_name(arg, name);
}

template <typename T>
void testing_gemm(const Arguments& arg)
{
    bool FORTRAN       = arg.fortran;
    auto hipblasGemmFn = FORTRAN ? hipblasGemm<T, true> : hipblasGemm<T, false>;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB = char2hipblas_operation(arg.transB);
    int                M      = arg.M;
    int                N      = arg.N;
    int                K      = arg.K;
    int                lda    = arg.lda;
    int                ldb    = arg.ldb;
    int                ldc    = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

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

    size_t A_size = size_t(lda) * A_col;
    size_t B_size = size_t(ldb) * B_col;
    size_t C_size = size_t(ldc) * N;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        return;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_copy(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, A_row, A_col, lda, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, arg, B_row, B_col, ldb, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_matrix(hC_host, arg, M, N, ldc, 0, 1, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy   = hC_host;
    hC_device = hC_host;

    // copy data from CPU to device, does not work for lda != A_row
    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA, sizeof(T) * lda * A_col, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dB, hB, sizeof(T) * ldb * B_col, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dC, hC_host, sizeof(T) * ldc * N, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        // library interface
        ASSERT_HIPBLAS_SUCCESS(hipblasGemmFn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        // copy output from device to CPU
        ASSERT_HIP_SUCCESS(hipMemcpy(hC_host, dC, sizeof(T) * ldc * N, hipMemcpyDeviceToHost));

        ASSERT_HIP_SUCCESS(hipMemcpy(dC, hC_device, sizeof(T) * ldc * N, hipMemcpyHostToDevice));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasGemmFn(
            handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        ASSERT_HIP_SUCCESS(hipMemcpy(hC_device, dC, sizeof(T) * ldc * N, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_gemm<T>(transA,
                      transB,
                      M,
                      N,
                      K,
                      h_alpha,
                      hA.data(),
                      lda,
                      hB.data(),
                      ldb,
                      h_beta,
                      hC_copy.data(),
                      ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            if(std::is_same_v<T, hipblasHalf> && (getArchMajor() == 11))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                near_check_general<T>(M, N, ldc, hC_copy.data(), hC_host.data(), tol);
                near_check_general<T>(M, N, ldc, hC_copy.data(), hC_device.data(), tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_copy, hC_host);
                unit_check_general<T>(M, N, ldc, hC_copy, hC_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host = std::abs(norm_check_general<T>('F', M, N, ldc, hC_copy, hC_host));
            hipblas_error_device
                = std::abs(norm_check_general<T>('F', M, N, ldc, hC_copy, hC_device));
        }

    } // end of if unit/norm check

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

            ASSERT_HIPBLAS_SUCCESS(hipblasGemmFn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       gemm_gflop_count<T>(M, N, K),
                                       gemm_gbyte_count<T>(M, N, K),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
