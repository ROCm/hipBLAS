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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using hipblasIamaxIaminModel = ArgumentModel<e_N, e_incx>;

template <typename T>
using hipblas_iamax_iamin_t
    = hipblasStatus_t (*)(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
inline hipblasStatus_t testing_iamax_iamin(const Arguments& arg, hipblas_iamax_iamin_t<T> func)
{
    int N    = arg.N;
    int incx = arg.incx;

    hipblasLocalHandle handle(arg);

    int zero = 0;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<int> d_hipblas_result_0(1);
        host_vector<int>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(int), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, nullptr, incx, d_hipblas_result_0));

        host_vector<int> cpu_0(1);
        host_vector<int> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(int), hipMemcpyDeviceToHost));
        unit_check_general<int>(1, 1, 1, cpu_0, gpu_0);

        return HIPBLAS_STATUS_SUCCESS;
    }

    size_t sizeX = size_t(N) * incx;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this
    // practice
    host_vector<T> hx(sizeX);
    int            cpu_result, hipblas_result_host, hipblas_result_device;

    device_vector<T>   dx(sizeX);
    device_vector<int> d_hipblas_result(1);

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, N, incx, 0, 1, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice));

    double gpu_time_used;
    int    hipblas_error_host, hipblas_error_device;

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // device_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, d_hipblas_result));

        CHECK_HIP_ERROR(hipMemcpy(
            &hipblas_result_device, d_hipblas_result, sizeof(int), hipMemcpyDeviceToHost));

        // host_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, &hipblas_result_host));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        REFBLAS_FUNC(N, hx.data(), incx, &cpu_result);
        // change to Fortran 1 based indexing as in BLAS standard, not cblas zero based indexing
        cpu_result += 1;

        if(arg.unit_check)
        {
            unit_check_general<int>(1, 1, 1, &cpu_result, &hipblas_result_host);
            unit_check_general<int>(1, 1, 1, &cpu_result, &hipblas_result_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host   = std::abs(hipblas_result_host - cpu_result);
            hipblas_error_device = std::abs(hipblas_result_device - cpu_result);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(func(handle, N, dx, incx, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasIamaxIaminModel{}.log_args<T>(std::cout,
                                             arg,
                                             gpu_time_used,
                                             iamax_gflop_count<T>(N),
                                             iamax_gbyte_count<T>(N),
                                             hipblas_error_host,
                                             hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

inline void testname_amax(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_amax(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasIamaxFn = FORTRAN ? hipblasIamax<T, true> : hipblasIamax<T, false>;

    return testing_iamax_iamin<T, cblas_iamax<T>>(arg, hipblasIamaxFn);
}

inline void testname_amin(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_amin(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasIaminFn = FORTRAN ? hipblasIamin<T, true> : hipblasIamin<T, false>;

    return testing_iamax_iamin<T, cblas_iamin<T>>(arg, hipblasIamin<T>);
}
