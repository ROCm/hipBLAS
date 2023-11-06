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

using hipblasGemvModel
    = ArgumentModel<e_a_type, e_transA, e_M, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy>;

inline void testname_gemv(const Arguments& arg, std::string& name)
{
    hipblasGemvModel{}.test_name(arg, name);
}

template <typename T>
void testing_gemv(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemvFn = FORTRAN ? hipblasGemv<T, true> : hipblasGemv<T, false>;

    int M    = arg.M;
    int N    = arg.N;
    int lda  = arg.lda;
    int incx = arg.incx;
    int incy = arg.incy;

    size_t A_size = size_t(lda) * N;
    size_t X_size, dim_x;
    size_t Y_size, dim_y;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);

    if(transA == HIPBLAS_OP_N)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy;
    if(invalid_size || !M || !N)
    {
        // Only rocBLAS conforms to expected behaviour so commenting out
        /*
        hipblasStatus_t actual = hipblasGemvFn(
            handle, transA, M, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy);
        EXPECT_HIPBLAS_STATUS2(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
	*/
        return;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;
    X_size       = dim_x * abs_incx;
    Y_size       = dim_y * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hy(Y_size);
    host_vector<T> hy_cpu(Y_size);
    host_vector<T> hy_host(Y_size);
    host_vector<T> hy_device(Y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dy(Y_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, lda, N, lda, 0, 1, hipblas_client_alpha_sets_nan, true, false);
    hipblas_init_vector(hx, arg, dim_x, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, dim_y, abs_incy, 0, 1, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    if(arg.unit_check || arg.norm_check)
    {
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasGemvFn(
            handle, transA, M, N, (T*)&h_alpha, dA, lda, dx, incx, (T*)&h_beta, dy, incy));

        ASSERT_HIP_SUCCESS(
            hipMemcpy(hy_host.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));
        ASSERT_HIP_SUCCESS(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(
            hipblasGemvFn(handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));

        ASSERT_HIP_SUCCESS(
            hipMemcpy(hy_device.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_gemv<T>(
            transA, M, N, h_alpha, hA.data(), lda, hx.data(), incx, h_beta, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_cpu, hy_host);
            unit_check_general<T>(1, dim_y, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_cpu, hy_host);
            hipblas_error_device
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_cpu, hy_device);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIP_SUCCESS(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
            {
                gpu_time_used = get_time_us_sync(stream);
            }

            ASSERT_HIPBLAS_SUCCESS(
                hipblasGemvFn(handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemvModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       gemv_gflop_count<T>(transA, M, N),
                                       gemv_gbyte_count<T>(transA, M, N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
