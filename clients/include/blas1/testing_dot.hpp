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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasDotModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy>;

inline void testname_dot(const Arguments& arg, std::string& name)
{
    hipblasDotModel{}.test_name(arg, name);
}

inline void testname_dotc(const Arguments& arg, std::string& name)
{
    hipblasDotModel{}.test_name(arg, name);
}

template <typename T, bool CONJ = false>
void testing_dot(const Arguments& arg)
{
    bool FORTRAN      = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDotFn = FORTRAN ? (CONJ ? hipblasDotc<T, true> : hipblasDot<T, true>)
                                : (CONJ ? hipblasDotc<T, false> : hipblasDot<T, false>);

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        device_vector<T> d_hipblas_result_0(1);
        host_vector<T>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        ASSERT_HIP_SUCCESS(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(T), hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(
            hipblasDotFn(handle, N, nullptr, incx, nullptr, incy, d_hipblas_result_0));

        host_vector<T> cpu_0(1);
        host_vector<T> gpu_0(1);

        ASSERT_HIP_SUCCESS(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(T), hipMemcpyDeviceToHost));
        unit_check_general<T>(1, 1, 1, cpu_0, gpu_0);

        return;
    }

    int    abs_incx = incx >= 0 ? incx : -incx;
    int    abs_incy = incy >= 0 ? incy : -incy;
    size_t sizeX    = size_t(N) * abs_incx;
    size_t sizeY    = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);

    T                cpu_result, h_hipblas_result_1, h_hipblas_result_2;
    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);
    device_vector<T> d_hipblas_result(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(hy, arg, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);

    // copy data from CPU to device, does not work for incx != 1
    ASSERT_HIP_SUCCESS(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        // hipblasDot accept both dev/host pointer for the scalar
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS((hipblasDotFn)(handle, N, dx, incx, dy, incy, d_hipblas_result));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS((hipblasDotFn)(handle, N, dx, incx, dy, incy, &h_hipblas_result_1));

        ASSERT_HIP_SUCCESS(
            hipMemcpy(&h_hipblas_result_2, d_hipblas_result, sizeof(T), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        if(arg.unit_check)
        {
            unit_check_general<T>(1, 1, 1, &cpu_result, &h_hipblas_result_1);
            unit_check_general<T>(1, 1, 1, &cpu_result, &h_hipblas_result_2);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, 1, 1, &cpu_result, &h_hipblas_result_1);
            hipblas_error_device
                = norm_check_general<T>('F', 1, 1, 1, &cpu_result, &h_hipblas_result_2);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS((hipblasDotFn)(handle, N, dx, incx, dy, incy, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotModel{}.log_args<T>(std::cout,
                                      arg,
                                      gpu_time_used,
                                      dot_gflop_count<CONJ, T>(N),
                                      dot_gbyte_count<T>(N),
                                      hipblas_error_host,
                                      hipblas_error_device);
    }
}

template <typename T>
void testing_dotc(const Arguments& arg)
{
    testing_dot<T, true>(arg);
}

template <typename T, bool CONJ = false>
hipblasStatus_t testing_dot_ret(const Arguments& arg)
{
    testing_dot<T, CONJ>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_dotc_ret(const Arguments& arg)
{
    testing_dotc<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
