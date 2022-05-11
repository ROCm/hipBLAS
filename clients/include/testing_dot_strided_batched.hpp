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

/* ============================================================================================ */

template <typename T, bool CONJ = false>
hipblasStatus_t testing_dot_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDotStridedBatchedFn
        = FORTRAN
              ? (CONJ ? hipblasDotcStridedBatched<T, true> : hipblasDotStridedBatched<T, true>)
              : (CONJ ? hipblasDotcStridedBatched<T, false> : hipblasDotStridedBatched<T, false>);

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int           abs_incx = incx >= 0 ? incx : -incx;
    int           abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stridex  = size_t(N) * abs_incx * stride_scale;
    hipblasStride stridey  = size_t(N) * abs_incy * stride_scale;
    size_t        sizeX    = stridex * batch_count;
    size_t        sizeY    = stridey * batch_count;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<T> d_hipblas_result_0(std::max(batch_count, 1));
        host_vector<T>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(d_hipblas_result_0,
                                  h_hipblas_result_0,
                                  sizeof(T) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotStridedBatchedFn(handle,
                                                       N,
                                                       nullptr,
                                                       incx,
                                                       stridex,
                                                       nullptr,
                                                       incy,
                                                       stridey,
                                                       batch_count,
                                                       d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<T> cpu_0(batch_count);
            host_vector<T> gpu_0(batch_count);

            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(T) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<T>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> h_hipblas_result1(batch_count);
    host_vector<T> h_hipblas_result2(batch_count);
    host_vector<T> h_cpu_result(batch_count);

    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);
    device_vector<T> d_hipblas_result(batch_count);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(
        hx, argus, N, abs_incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(
        hy, argus, N, abs_incy, stridey, batch_count, hipblas_client_alpha_sets_nan, false);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        // hipblasDot accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR((hipblasDotStridedBatchedFn)(handle,
                                                         N,
                                                         dx,
                                                         incx,
                                                         stridex,
                                                         dy,
                                                         incy,
                                                         stridey,
                                                         batch_count,
                                                         d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR((hipblasDotStridedBatchedFn)(handle,
                                                         N,
                                                         dx,
                                                         incx,
                                                         stridex,
                                                         dy,
                                                         incy,
                                                         stridey,
                                                         batch_count,
                                                         h_hipblas_result1));

        CHECK_HIP_ERROR(hipMemcpy(
            h_hipblas_result2, d_hipblas_result, sizeof(T) * batch_count, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N,
                                                  hx.data() + b * stridex,
                                                  incx,
                                                  hy.data() + b * stridey,
                                                  incy,
                                                  &h_cpu_result[b]);
        }

        if(argus.unit_check)
        {
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_hipblas_result1);
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_hipblas_result2);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, batch_count, 1, h_cpu_result, h_hipblas_result1);
            hipblas_error_device
                = norm_check_general<T>('F', 1, batch_count, 1, h_cpu_result, h_hipblas_result2);
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR((hipblasDotStridedBatchedFn)(handle,
                                                             N,
                                                             dx,
                                                             incx,
                                                             stridex,
                                                             dy,
                                                             incy,
                                                             stridey,
                                                             batch_count,
                                                             d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_incy, e_stride_y, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            dot_gflop_count<CONJ, T>(N),
            dot_gbyte_count<T>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_dotc_strided_batched(const Arguments& argus)
{
    return testing_dot_strided_batched<T, true>(argus);
}
