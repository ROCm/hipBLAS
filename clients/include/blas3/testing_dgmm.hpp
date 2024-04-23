/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasDgmmModel = ArgumentModel<e_a_type, e_side, e_M, e_N, e_lda, e_incx, e_ldc>;

inline void testname_dgmm(const Arguments& arg, std::string& name)
{
    hipblasDgmmModel{}.test_name(arg, name);
}

template <typename T>
void testing_dgmm_bad_arg(const Arguments& arg)
{
    auto hipblasDgmmFn = arg.api == FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;
    auto hipblasDgmmFn_64
        = arg.api == FORTRAN_64 ? hipblasDgmm_64<T, true> : hipblasDgmm_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M    = 101;
    int64_t N    = 100;
    int64_t lda  = 102;
    int64_t incx = 1;
    int64_t ldc  = 103;

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    device_vector<T> dA(N * lda);
    device_vector<T> dx(incx * K);
    device_vector<T> dC(N * ldc);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasDgmmFn,
                (nullptr, side, M, N, dA, lda, dx, incx, dC, ldc));

    DAPI_EXPECT(
        HIPBLAS_STATUS_INVALID_ENUM,
        hipblasDgmmFn,
        (handle, (hipblasSideMode_t)HIPBLAS_FILL_MODE_FULL, M, N, dA, lda, dx, incx, dC, ldc));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle, side, M, N, nullptr, lda, dx, incx, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle, side, M, N, dA, lda, nullptr, incx, dC, ldc));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle, side, M, N, dA, lda, dx, incx, nullptr, ldc));

        // dgmm will quick-return with M == 0 || N == 0. Here, c_i32_overflow will rollover in the case of 32-bit params,
        // and quick-return with 64-bit params. This depends on implementation so only testing rocBLAS backend
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle,
                     side,
                     0,
                     c_i32_overflow,
                     nullptr,
                     c_i32_overflow,
                     nullptr,
                     incx,
                     nullptr,
                     c_i32_overflow));
        DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasDgmmFn,
                    (handle,
                     side,
                     c_i32_overflow,
                     0,
                     nullptr,
                     c_i32_overflow,
                     nullptr,
                     incx,
                     nullptr,
                     c_i32_overflow));
    }

    // If M == 0 || N == 0, can have nullptrs
    DAPI_CHECK(hipblasDgmmFn, (handle, side, 0, N, nullptr, lda, nullptr, incx, nullptr, ldc));
    DAPI_CHECK(hipblasDgmmFn, (handle, side, M, 0, nullptr, lda, nullptr, incx, nullptr, ldc));
}

template <typename T>
void testing_dgmm(const Arguments& arg)
{
    auto hipblasDgmmFn = arg.api == FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;
    auto hipblasDgmmFn_64
        = arg.api == FORTRAN_64 ? hipblasDgmm_64<T, true> : hipblasDgmm_64<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);

    int64_t M    = arg.M;
    int64_t N    = arg.N;
    int64_t lda  = arg.lda;
    int64_t incx = arg.incx;
    int64_t ldc  = arg.ldc;

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    size_t  A_size   = lda * N;
    size_t  C_size   = ldc * N;
    int64_t k        = (side == HIPBLAS_SIDE_RIGHT ? N : M);
    size_t  X_size   = abs_incx * k;
    if(!X_size)
        X_size = 1;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M;
    if(invalid_size || !N || !M)
    {
        DAPI_EXPECT((invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS),
                    hipblasDgmmFn,
                    (handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc));
        return;
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
    hipblas_init_matrix(hA, arg, M, N, lda, 0, 1, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, arg, k, abs_incx, 0, 1, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(hC, arg, M, N, ldc, 0, 1, hipblas_client_never_set_nan);
    hA_copy = hA;
    hx_copy = hx;
    hC_1    = hC;
    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasDgmmFn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        ref_dgmm<T>(side, M, N, hA_copy, lda, hx_copy, incx, hC_gold, ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
        }

        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
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

            DAPI_DISPATCH(hipblasDgmmFn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasDgmmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       dgmm_gflop_count<T>(M, N),
                                       dgmm_gbyte_count<T>(M, N, k),
                                       hipblas_error);
    }
}
