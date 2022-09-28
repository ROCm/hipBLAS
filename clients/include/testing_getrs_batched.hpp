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

using hipblasGetrsBatchedModel = ArgumentModel<e_N, e_lda, e_ldb, e_batch_count>;

inline void testname_getrs_batched(const Arguments& arg, std::string& name)
{
    hipblasGetrsBatchedModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_getrs_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.fortran;
    auto hipblasGetrsBatchedFn
        = FORTRAN ? hipblasGetrsBatched<T, true> : hipblasGetrsBatched<T, false>;

    int N           = arg.N;
    int lda         = arg.lda;
    int ldb         = arg.ldb;
    int batch_count = arg.batch_count;

    hipblasStride strideP   = N;
    size_t        A_size    = size_t(lda) * N;
    size_t        B_size    = size_t(ldb) * 1;
    size_t        Ipiv_size = strideP * batch_count;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hX(B_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hB1(B_size, 1, batch_count);
    host_vector<int>     hIpiv(Ipiv_size);
    host_vector<int>     hIpiv1(Ipiv_size);
    int                  info;

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_vector<int>     dIpiv(Ipiv_size);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    // Initial hA, hB, hX on CPU
    hipblas_init(hA, true);
    hipblas_init(hX);
    srand(1);
    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        // scale A to avoid singularities
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }

        // Calculate hB = hA*hX;
        cblas_gemm<T>(op, op, N, 1, N, (T)1, hA[b], lda, hX[b], ldb, (T)0, hB[b], ldb);

        // LU factorize hA on the CPU
        info = cblas_getrf<T>(N, N, hA[b], lda, hIpiv.data() + b * strideP);
        if(info != 0)
        {
            std::cerr << "LU decomposition failed" << std::endl;
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), Ipiv_size * sizeof(int), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrsBatchedFn(handle,
                                                  op,
                                                  N,
                                                  1,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dIpiv,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &info,
                                                  batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB1.transfer_from(dB));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_getrs('N', N, 1, hA[b], lda, hIpiv.data() + b * strideP, hB[b], ldb);
        }

        hipblas_error = norm_check_general<T>('F', N, 1, ldb, hB, hB1, batch_count);
        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;

            unit_check_error(hipblas_error, tolerance);
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

            CHECK_HIPBLAS_ERROR(hipblasGetrsBatchedFn(handle,
                                                      op,
                                                      N,
                                                      1,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dIpiv,
                                                      dB.ptr_on_device(),
                                                      ldb,
                                                      &info,
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGetrsBatchedModel{}.log_args<T>(std::cout,
                                               arg,
                                               gpu_time_used,
                                               getrs_gflop_count<T>(N, 1),
                                               ArgumentLogging::NA_value,
                                               hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
