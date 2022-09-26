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
#include <vector>

#include "testing_common.hpp"

// strides not used
using hipblasGeqrfStridedBatchedModel = ArgumentModel<e_M, e_N, e_lda, e_batch_count>;

inline void testname_geqrf_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGeqrfStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_geqrf_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.fortran;
    auto hipblasGeqrfStridedBatchedFn
        = FORTRAN ? hipblasGeqrfStridedBatched<T, true> : hipblasGeqrfStridedBatched<T, false>;

    int    M            = arg.M;
    int    N            = arg.N;
    int    K            = std::min(M, N);
    int    lda          = arg.lda;
    int    batch_count  = arg.batch_count;
    double stride_scale = arg.stride_scale;

    hipblasStride strideA   = lda * N * stride_scale;
    hipblasStride strideP   = K * stride_scale;
    int           A_size    = strideA * batch_count;
    int           Ipiv_size = strideP * batch_count;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || lda < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA1(A_size);
    host_vector<T> hIpiv(Ipiv_size);
    host_vector<T> hIpiv1(Ipiv_size);
    int            info;

    device_vector<T> dA(A_size);
    device_vector<T> dIpiv(Ipiv_size);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    // Initial hA on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        T* hAb = hA.data() + b * strideA;

        hipblas_init<T>(hAb, M, N, lda);

        // scale A to avoid singularities
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hAb[i + j * lda] += 400;
                else
                    hAb[i + j * lda] -= 4;
            }
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(T)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasGeqrfStridedBatchedFn(
        handle, M, N, dA, lda, strideA, dIpiv, strideP, &info, batch_count));

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1.data(), dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        cblas_geqrf(M, N, hA.data(), lda, hIpiv.data(), work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        for(int b = 0; b < batch_count; b++)
        {
            cblas_geqrf(
                M, N, hA.data() + b * strideA, lda, hIpiv.data() + b * strideP, work.data(), N);
        }

        double e1     = norm_check_general<T>('F', M, N, lda, strideA, hA, hA1, batch_count);
        double e2     = norm_check_general<T>('F', K, 1, K, strideP, hIpiv, hIpiv1, batch_count);
        hipblas_error = e1 + e2;

        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(e1, tolerance);
            unit_check_error(e2, tolerance);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGeqrfStridedBatchedFn(
                handle, M, N, dA, lda, strideA, dIpiv, strideP, &info, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGeqrfStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      geqrf_gflop_count<T>(N, M),
                                                      ArgumentLogging::NA_value,
                                                      hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
