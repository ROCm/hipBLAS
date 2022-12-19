/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasGemmStridedBatchedModel = ArgumentModel<e_transA,
                                                     e_transB,
                                                     e_M,
                                                     e_N,
                                                     e_K,
                                                     e_alpha,
                                                     e_lda,
                                                     e_ldb,
                                                     e_beta,
                                                     e_ldc,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_gemm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGemmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_gemm_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasGemmStridedBatchedFn
        = FORTRAN ? hipblasGemmStridedBatched<T, true> : hipblasGemmStridedBatched<T, false>;

    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB       = char2hipblas_operation(arg.transB);
    int                M            = arg.M;
    int                N            = arg.N;
    int                K            = arg.K;
    int                lda          = arg.lda;
    int                ldb          = arg.ldb;
    int                ldc          = arg.ldc;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
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

    hipblasStride stride_A = size_t(lda) * A_col * stride_scale;
    hipblasStride stride_B = size_t(ldb) * B_col * stride_scale;
    hipblasStride stride_C = size_t(ldc) * N * stride_scale;
    size_t        A_size   = stride_A * batch_count;
    size_t        B_size   = stride_B * batch_count;
    size_t        C_size   = stride_C * batch_count;

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
    hipblas_init_matrix(
        hA, arg, A_row, A_col, lda, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, arg, B_row, B_col, ldb, stride_B, batch_count, hipblas_client_alpha_sets_nan);
    hipblas_init_matrix(
        hC_host, arg, M, N, ldc, stride_C, batch_count, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy   = hC_host;
    hC_device = hC_host;

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        // host mode
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        // library interface
        CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
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
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        // device mode
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
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
        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T>(transA,
                          transB,
                          M,
                          N,
                          K,
                          h_alpha,
                          hA.data() + stride_A * i,
                          lda,
                          hB.data() + stride_B * i,
                          ldb,
                          h_beta,
                          hC_copy.data() + stride_C * i,
                          ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        // gemm has better performance in host mode. In rocBLAS in device mode
        // we need to copy alpha and beta to the host.
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
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
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     gemm_gflop_count<T>(M, N, K),
                                                     gemm_gbyte_count<T>(M, N, K),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
