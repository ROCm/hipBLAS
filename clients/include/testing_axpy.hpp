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

using hipblasAxpyModel = ArgumentModel<e_N, e_alpha, e_incx, e_incy>;

inline void testname_axpy(const Arguments& arg, std::string& name)
{
    hipblasAxpyModel{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_axpy(const Arguments& arg)
{
    bool FORTRAN       = arg.fortran;
    auto hipblasAxpyFn = FORTRAN ? hipblasAxpy<T, true> : hipblasAxpy<T, false>;

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasAxpyFn(handle, N, nullptr, nullptr, incx, nullptr, incy));
        return HIPBLAS_STATUS_SUCCESS;
    }

    size_t sizeX = size_t(N) * abs_incx;
    size_t sizeY = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    T alpha = arg.get_alpha<T>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy_host(sizeY);
    host_vector<T> hy_device(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    device_vector<T> dx(sizeX);
    device_vector<T> dy_host(sizeY);
    device_vector<T> dy_device(sizeY);
    device_vector<T> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, arg, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);
    hy_device = hy_host;

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy_host;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_host, hy_host.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dy_device, hy_device.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasAxpyFn(handle, N, d_alpha, dx, incx, dy_device, incy));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasAxpyFn(handle, N, &alpha, dx, incx, dy_host, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hy_host.data(), dy_host, sizeof(T) * sizeY, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hy_device.data(), dy_device, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_axpy<T>(N, alpha, hx_cpu.data(), incx, hy_cpu.data(), incy);

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy_host.data());
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu.data(), hy_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu.data(), hy_device.data());
        }

    } // end of if unit check

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

            CHECK_HIPBLAS_ERROR(hipblasAxpyFn(handle, N, d_alpha, dx, incx, dy_device, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       axpy_gflop_count<T>(N),
                                       axpy_gbyte_count<T>(N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
