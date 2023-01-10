/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasTpsvModel = ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_incx>;

inline void testname_tpsv(const Arguments& arg, std::string& name)
{
    hipblasTpsvModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_tpsv(const Arguments& arg)
{
    bool FORTRAN       = arg.fortran;
    auto hipblasTpsvFn = FORTRAN ? hipblasTpsv<T, true> : hipblasTpsv<T, false>;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    int                N      = arg.N;
    int                incx   = arg.incx;

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t size_A   = size_t(N) * N;
    size_t size_AP  = size_t(N) * (N + 1) / 2;
    size_t size_x   = abs_incx * size_t(N);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx;
    if(invalid_size || !N)
    {
        hipblasStatus_t actual
            = hipblasTpsvFn(handle, uplo, transA, diag, N, nullptr, nullptr, incx);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hAP(size_AP);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    device_vector<T> dAP(size_AP);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    // srand(1);
    // hipblas_init<T>(hA, N, N, 1);
    // hipblas_init<T>(hx, 1, N, abs_incx);
    hipblas_init_matrix(hA, arg, size_A, 1, 1, 0, 1, hipblas_client_never_set_nan, true, false);
    hipblas_init_vector(
        hx, arg, N, abs_incx, 0, 1, hipblas_client_never_set_nan, false, false); //true);
    hb = hx;

    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T>(HIPBLAS_OP_N,
                  HIPBLAS_OP_T,
                  N,
                  N,
                  N,
                  (T)1.0,
                  hA.data(),
                  N,
                  hA.data(),
                  N,
                  (T)0.0,
                  AAT.data(),
                  N);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * N] = AAT[i + j * N];
            t += std::abs(AAT[i + j * N]);
        }
        hA[i + i * N] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(arg.uplo, N, hA.data(), N);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(arg.diag == 'U' || arg.diag == 'u')
    {
        if('L' == arg.uplo || 'l' == arg.uplo)
            for(int i = 0; i < N; i++)
            {
                T diag = hA[i + i * N];
                for(int j = 0; j <= i; j++)
                    hA[i + j * N] = hA[i + j * N] / diag;
            }
        else
            for(int j = 0; j < N; j++)
            {
                T diag = hA[j + j * N];
                for(int i = 0; i <= j; i++)
                    hA[i + j * N] = hA[i + j * N] / diag;
            }
    }

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, N, hA.data(), N, hb.data(), incx);
    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA, (T*)hAP, N);

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dAP, hAP.data(), sizeof(T) * size_AP, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasTpsvFn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // Calculating error
        hipblas_error = std::abs(vector_norm_1<T>(N, abs_incx, hx.data(), hx_or_b_1.data()));

        if(arg.unit_check)
        {
            double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
            unit_check_error(hipblas_error, tolerance);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTpsvFn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTpsvModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       tpsv_gflop_count<T>(N),
                                       tpsv_gbyte_count<T>(N),
                                       hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
