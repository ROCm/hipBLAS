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

inline void testname_getrs_strided_batched(const Arguments& arg, std::string& name)
{
    ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_getrs_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.fortran;
    auto hipblasGetrsStridedBatchedFn
        = FORTRAN ? hipblasGetrsStridedBatched<T, true> : hipblasGetrsStridedBatched<T, false>;

    int    N            = arg.N;
    int    lda          = arg.lda;
    int    ldb          = arg.ldb;
    int    batch_count  = arg.batch_count;
    double stride_scale = arg.stride_scale;

    hipblasStride strideA   = size_t(lda) * N * stride_scale;
    hipblasStride strideB   = size_t(ldb) * 1 * stride_scale;
    hipblasStride strideP   = size_t(N) * stride_scale;
    size_t        A_size    = strideA * batch_count;
    size_t        B_size    = strideB * batch_count;
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
    host_vector<T>   hA(A_size);
    host_vector<T>   hX(B_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hB1(B_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    int              info;

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dIpiv(Ipiv_size);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        T*   hAb    = hA.data() + b * strideA;
        T*   hXb    = hX.data() + b * strideB;
        T*   hBb    = hB.data() + b * strideB;
        int* hIpivb = hIpiv.data() + b * strideP;

        hipblas_init<T>(hAb, N, N, lda);
        hipblas_init<T>(hXb, N, 1, ldb);

        // scale A to avoid singularities
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hAb[i + j * lda] += 400;
                else
                    hAb[i + j * lda] -= 4;
            }
        }

        // Calculate hB = hA*hX;
        cblas_gemm<T>(op, op, N, 1, N, (T)1, hAb, lda, hXb, ldb, (T)0, hBb, ldb);

        // LU factorize hA on the CPU
        info = cblas_getrf<T>(N, N, hAb, lda, hIpivb);
        if(info != 0)
        {
            std::cerr << "LU decomposition failed" << std::endl;
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, B_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv, Ipiv_size * sizeof(int), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrsStridedBatchedFn(handle,
                                                         op,
                                                         N,
                                                         1,
                                                         dA,
                                                         lda,
                                                         strideA,
                                                         dIpiv,
                                                         strideP,
                                                         dB,
                                                         ldb,
                                                         strideB,
                                                         &info,
                                                         batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB1.data(), dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_getrs('N',
                        N,
                        1,
                        hA.data() + b * strideA,
                        lda,
                        hIpiv.data() + b * strideP,
                        hB.data() + b * strideB,
                        ldb);
        }

        hipblas_error = norm_check_general<T>('F', N, 1, ldb, strideB, hB, hB1, batch_count);

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

            CHECK_HIPBLAS_ERROR(hipblasGetrsStridedBatchedFn(handle,
                                                             op,
                                                             N,
                                                             1,
                                                             dA,
                                                             lda,
                                                             strideA,
                                                             dIpiv,
                                                             strideP,
                                                             dB,
                                                             ldb,
                                                             strideB,
                                                             &info,
                                                             batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_lda, e_stride_a, e_ldb, e_stride_b, e_batch_count>{}.log_args<T>(
            std::cout,
            arg,
            gpu_time_used,
            getrs_gflop_count<T>(N, 1),
            ArgumentLogging::NA_value,
            hipblas_error);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
