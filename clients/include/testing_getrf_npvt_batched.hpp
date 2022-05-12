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

template <typename T>
hipblasStatus_t testing_getrf_npvt_batched(const Arguments& argus)
{
    using U      = real_t<T>;
    bool FORTRAN = argus.fortran;
    auto hipblasGetrfBatchedFn
        = FORTRAN ? hipblasGetrfBatched<T, true> : hipblasGetrfBatched<T, false>;

    int M           = argus.N;
    int N           = argus.N;
    int lda         = argus.lda;
    int batch_count = argus.batch_count;

    hipblasStride strideP   = std::min(M, N);
    size_t        A_size    = size_t(lda) * N;
    size_t        Ipiv_size = strideP * batch_count;

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
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hA1(A_size, 1, batch_count);
    host_vector<int>     hIpiv(Ipiv_size);
    host_vector<int>     hInfo(batch_count);
    host_vector<int>     hInfo1(batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_vector<int>     dInfo(batch_count);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial hA on CPU
    hipblas_init(hA, true);
    for(int b = 0; b < batch_count; b++)
    {
        // scale A to avoid singularities
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }
    }

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, batch_count * sizeof(int)));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(
            hipblasGetrfBatchedFn(handle, N, dA.ptr_on_device(), lda, nullptr, dInfo, batch_count));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(hA1.transfer_from(dA));
        CHECK_HIP_ERROR(
            hipMemcpy(hInfo1.data(), dInfo, batch_count * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            hInfo[b] = cblas_getrf(M, N, hA[b], lda, hIpiv.data() + b * strideP);
        }

        hipblas_error = norm_check_general<T>('F', M, N, lda, hA, hA1, batch_count);
        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(hipblas_error, tolerance);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGetrfBatchedFn(
                handle, N, dA.ptr_on_device(), lda, nullptr, dInfo, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_lda, e_batch_count>{}.log_args<T>(std::cout,
                                                               argus,
                                                               gpu_time_used,
                                                               getrf_gflop_count<T>(N, M),
                                                               ArgumentLogging::NA_value,
                                                               hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
