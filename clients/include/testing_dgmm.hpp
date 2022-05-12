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

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_dgmm(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasDgmmFn = FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;

    hipblasSideMode_t side = char2hipblas_side(argus.side_option);

    int M    = argus.M;
    int N    = argus.N;
    int lda  = argus.lda;
    int incx = argus.incx;
    int ldc  = argus.ldc;

    int    abs_incx = incx >= 0 ? incx : -incx;
    size_t A_size   = size_t(lda) * N;
    size_t C_size   = size_t(ldc) * N;
    int    k        = (side == HIPBLAS_SIDE_RIGHT ? N : M);
    size_t X_size   = size_t(abs_incx) * k;
    if(!X_size)
        X_size = 1;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M;
    if(invalid_size || !N || !M)
    {
        hipblasStatus_t actual
            = hipblasDgmmFn(handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_copy(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hx_copy(X_size);
    host_vector<T> hC(C_size);
    host_vector<T> hC_1(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dC(C_size);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    hipblas_init_matrix(hA, argus, M, N, lda, 0, 1, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, argus, k, abs_incx, 0, 1, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(hC, argus, M, N, ldc, 0, 1, hipblas_client_never_set_nan);
    hA_copy = hA;
    hx_copy = hx;
    hC_1    = hC;
    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasDgmmFn(handle, side, M, N, dA, lda, dx, incx, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        for(size_t i1 = 0; i1 < M; i1++)
        {
            for(size_t i2 = 0; i2 < N; i2++)
            {
                if(HIPBLAS_SIDE_RIGHT == side)
                {
                    hC_gold[i1 + i2 * ldc] = hA_copy[i1 + i2 * lda] * hx_copy[shift_x + i2 * incx];
                }
                else
                {
                    hC_gold[i1 + i2 * ldc] = hA_copy[i1 + i2 * lda] * hx_copy[shift_x + i1 * incx];
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
        }

        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
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

            CHECK_HIPBLAS_ERROR(hipblasDgmmFn(handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_side_option, e_M, e_N, e_lda, e_incx, e_ldc>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            dgmm_gflop_count<T>(M, N),
            dgmm_gbyte_count<T>(M, N, k),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
