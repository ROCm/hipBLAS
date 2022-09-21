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
hipblasStatus_t testing_gels_strided_batched(const Arguments& argus)
{
    using U      = real_t<T>;
    bool FORTRAN = argus.fortran;
    auto hipblasGelsStridedBatchedFn
        = FORTRAN ? hipblasGelsStridedBatched<T, true> : hipblasGelsStridedBatched<T, false>;

    int    N           = argus.N;
    int    M           = argus.M;
    int    nrhs        = argus.K;
    int    lda         = argus.lda;
    int    ldb         = argus.ldb;
    char   transc      = argus.transA;
    int    batchCount  = argus.batch_count;
    double strideScale = argus.stride_scale;
    if(is_complex<T> && transc == 'T')
        transc = 'C';
    else if(!is_complex<T> && transc == 'C')
        transc = 'T';

    hipblasOperation_t trans = char2hipblas_operation(transc);

    hipblasStride strideA = size_t(lda) * N * strideScale;
    hipblasStride strideB = size_t(ldb) * nrhs * strideScale;
    size_t        A_size  = strideA * batchCount;
    size_t        B_size  = strideB * batchCount;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || nrhs < 0 || lda < M || ldb < M || ldb < N || batchCount < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batchCount == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA(A_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hB_res(B_size);
    host_vector<int> info_res(batchCount);
    host_vector<int> info(batchCount);
    int              info_input(-1);

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dInfo(batchCount);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblas_init<T>(hA, true);
    hipblas_init<T>(hB);
    for(int b = 0; b < batchCount; b++)
    {
        T* hAb = hA.data() + b * strideA;

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
    }

    hB_res = hB;

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, B_size * sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGelsStridedBatchedFn(handle,
                                                        trans,
                                                        M,
                                                        N,
                                                        nrhs,
                                                        dA,
                                                        lda,
                                                        strideA,
                                                        dB,
                                                        ldb,
                                                        strideB,
                                                        &info_input,
                                                        dInfo,
                                                        batchCount));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB_res, dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(info_res.data(), dInfo, sizeof(int) * batchCount, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        int            sizeW = std::max(1, std::min(M, N) + std::max(std::min(M, N), nrhs));
        host_vector<T> hW(sizeW);

        for(int b = 0; b < batchCount; b++)
        {
            info[b] = cblas_gels(transc,
                                 M,
                                 N,
                                 nrhs,
                                 hA.data() + b * strideA,
                                 lda,
                                 hB.data() + b * strideB,
                                 ldb,
                                 hW.data(),
                                 sizeW);
        }

        hipblas_error = norm_check_general<T>(
            'F', std::max(M, N), nrhs, ldb, strideB, hB.data(), hB_res.data(), batchCount);

        if(info_input != 0)
            hipblas_error += 1.0;
        for(int b = 0; b < batchCount; b++)
        {
            if(info[b] != info_res[b])
                hipblas_error += 1.0;
        }

        if(argus.unit_check)
        {
            double eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;

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

            CHECK_HIPBLAS_ERROR(hipblasGelsStridedBatchedFn(handle,
                                                            trans,
                                                            M,
                                                            N,
                                                            nrhs,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            dB,
                                                            ldb,
                                                            strideB,
                                                            &info_input,
                                                            dInfo,
                                                            batchCount));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_lda, e_ldb>{}.log_args<T>(std::cout,
                                                       argus,
                                                       gpu_time_used,
                                                       ArgumentLogging::NA_value,
                                                       ArgumentLogging::NA_value,
                                                       hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
