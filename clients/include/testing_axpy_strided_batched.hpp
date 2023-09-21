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

using hipblasAxpyStridedBatchedModel
    = ArgumentModel<e_N, e_alpha, e_incx, e_incy, e_stride_scale, e_batch_count>;

inline void testname_axpy_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasAxpyStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_axpy_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasAxpyStridedBatchedFn
        = FORTRAN ? hipblasAxpyStridedBatched<T, true> : hipblasAxpyStridedBatched<T, false>;

    int    N            = arg.N;
    int    incx         = arg.incx;
    int    incy         = arg.incy;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;
    T      alpha        = arg.get_alpha<T>();

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    hipblasStride stridex = size_t(N) * abs_incx * stride_scale;
    hipblasStride stridey = size_t(N) * abs_incy * stride_scale;
    size_t        sizeX   = stridex * batch_count;
    size_t        sizeY   = stridey * batch_count;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        ASSERT_HIPBLAS_SUCCESS(hipblasAxpyStridedBatchedFn(
            handle, N, nullptr, nullptr, incx, stridex, nullptr, incy, stridey, batch_count));
        return;
    }

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
    hipblas_init_vector(
        hx, arg, N, abs_incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(
        hy_host, arg, N, abs_incy, stridey, batch_count, hipblas_client_alpha_sets_nan, false);
    hy_device = hy_host;

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy_host;

    ASSERT_HIP_SUCCESS(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(
        hipMemcpy(dy_host, hy_host.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(
        hipMemcpy(dy_device, hy_device.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasAxpyStridedBatchedFn(
            handle, N, d_alpha, dx, incx, stridex, dy_device, incy, stridey, batch_count));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasAxpyStridedBatchedFn(
            handle, N, &alpha, dx, incx, stridex, dy_host, incy, stridey, batch_count));

        // copy output from device to CPU
        ASSERT_HIP_SUCCESS(
            hipMemcpy(hy_host.data(), dy_host, sizeof(T) * sizeY, hipMemcpyDeviceToHost));
        ASSERT_HIP_SUCCESS(
            hipMemcpy(hy_device.data(), dy_device, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy<T>(
                N, alpha, hx_cpu.data() + b * stridex, incx, hy_cpu.data() + b * stridey, incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(
                1, N, batch_count, abs_incy, stridex, hy_cpu.data(), hy_host.data());
            unit_check_general<T>(
                1, N, batch_count, abs_incy, stridey, hy_cpu.data(), hy_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, N, abs_incy, stridey, hy_cpu.data(), hy_host.data(), batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, N, abs_incy, stridey, hy_cpu.data(), hy_device.data(), batch_count);
        }
    } // end of if unit check

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

            ASSERT_HIPBLAS_SUCCESS(hipblasAxpyStridedBatchedFn(
                handle, N, d_alpha, dx, incx, stridex, dy_device, incy, stridey, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     axpy_gflop_count<T>(N),
                                                     axpy_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}

template <typename T>
hipblasStatus_t testing_axpy_strided_batched_ret(const Arguments& arg)
{
    testing_axpy_strided_batched<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
