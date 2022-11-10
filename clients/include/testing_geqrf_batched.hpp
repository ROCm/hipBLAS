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

using hipblasGeqrfBatchedModel = ArgumentModel<e_M, e_N, e_lda, e_batch_count>;

inline void testname_geqrf_batched(const Arguments& arg, std::string& name)
{
    hipblasGeqrfBatchedModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_geqrf_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.fortran;
    auto hipblasGeqrfBatchedFn
        = FORTRAN ? hipblasGeqrfBatched<T, true> : hipblasGeqrfBatched<T, false>;

    int M           = arg.M;
    int N           = arg.N;
    int K           = std::min(M, N);
    int lda         = arg.lda;
    int batch_count = arg.batch_count;

    size_t A_size    = size_t(lda) * N;
    int    Ipiv_size = K;
    int    info;

    hipblasLocalHandle handle(arg);

    // Check to prevent memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < std::max(1, M) || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        // including pointers so can test other params
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dIpiv(1, 1, 1);
        hipblasStatus_t        status = hipblasGeqrfBatched(
            handle, M, N, dA.ptr_on_device(), lda, dIpiv.ptr_on_device(), &info, batch_count);
        EXPECT_HIPBLAS_STATUS(
            status, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));

        int expected_info = 0;
        if(M < 0)
            expected_info = -1;
        else if(N < 0)
            expected_info = -2;
        else if(lda < std::max(1, M))
            expected_info = -4;
        else if(batch_count < 0)
            expected_info = -7;
        unit_check_general(1, 1, 1, &expected_info, &info);

        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hA1(A_size, 1, batch_count);
    host_batch_vector<T> hIpiv(Ipiv_size, 1, batch_count);
    host_batch_vector<T> hIpiv1(Ipiv_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dIpiv(Ipiv_size, 1, batch_count);

    double gpu_time_used, hipblas_error;

    // Initial hA on CPU
    hipblas_init(hA, true);
    srand(1);
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
    CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    CHECK_HIPBLAS_ERROR(hipblasGeqrfBatchedFn(
        handle, M, N, dA.ptr_on_device(), lda, dIpiv.ptr_on_device(), &info, batch_count));

    CHECK_HIP_ERROR(hIpiv1.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hA1.transfer_from(dA));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        cblas_geqrf(M, N, hA[0], lda, hIpiv[0], work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        for(int b = 0; b < batch_count; b++)
        {
            cblas_geqrf(M, N, hA[b], lda, hIpiv[b], work.data(), N);
        }

        double e1 = norm_check_general<T>('F', M, N, lda, hA, hA1, batch_count);
        double e2 = norm_check_general<T>('F', Ipiv_size, 1, Ipiv_size, hIpiv, hIpiv1, batch_count);
        hipblas_error = e1 + e2;

        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(e1, tolerance);
            unit_check_error(e2, tolerance);
            int zero = 0;
            unit_check_general(1, 1, 1, &zero, &info);
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

            CHECK_HIPBLAS_ERROR(hipblasGeqrfBatchedFn(
                handle, M, N, dA.ptr_on_device(), lda, dIpiv.ptr_on_device(), &info, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGeqrfBatchedModel{}.log_args<T>(std::cout,
                                               arg,
                                               gpu_time_used,
                                               geqrf_gflop_count<T>(N, M),
                                               ArgumentLogging::NA_value,
                                               hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
