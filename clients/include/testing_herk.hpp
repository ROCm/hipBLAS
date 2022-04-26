/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_herk(const Arguments& argus)
{
    using U            = real_t<T>;
    bool FORTRAN       = argus.fortran;
    auto hipblasHerkFn = FORTRAN ? hipblasHerk<T, U, true> : hipblasHerk<T, U, false>;

    int N   = argus.N;
    int K   = argus.K;
    int lda = argus.lda;
    int ldc = argus.ldc;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
       || (transA != HIPBLAS_OP_N && lda < K))
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int    K1     = (transA == HIPBLAS_OP_N ? K : N);
    size_t A_size = size_t(lda) * K1;
    size_t C_size = size_t(ldc) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dC(C_size);
    device_vector<U> d_alpha(1);
    device_vector<U> d_beta(1);

    U h_alpha = argus.get_alpha<U>();
    U h_beta  = argus.get_beta<U>();

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    hipblas_init_matrix(hA, argus, N, K1, lda, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(hC_host, argus, N, N, ldc, 0, 1, hipblas_client_beta_sets_nan, false, true);
    hC_device = hC_host;
    hC_gold   = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(U), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            hipblasHerkFn(handle, uplo, transA, N, K, &h_alpha, dA, lda, &h_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            hipblasHerkFn(handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_herk<T>(uplo, transA, N, K, h_alpha, hA, lda, h_beta, hC_gold, ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, ldc, hC_gold, hC_host);
            unit_check_general<T>(N, N, ldc, hC_gold, hC_device);
        }

        if(argus.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host);
            hipblas_error_device = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasHerkFn(handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option, e_transA_option, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         herk_gflop_count<T>(N, K),
                         herk_gbyte_count<T>(N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
