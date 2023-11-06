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

using hipblasDotBatchedModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy, e_batch_count>;

inline void testname_dot_batched(const Arguments& arg, std::string& name)
{
    hipblasDotBatchedModel{}.test_name(arg, name);
}

inline void testname_dotc_batched(const Arguments& arg, std::string& name)
{
    hipblasDotBatchedModel{}.test_name(arg, name);
}

template <typename T, bool CONJ = false>
void testing_dot_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDotBatchedFn
        = FORTRAN ? (CONJ ? hipblasDotcBatched<T, true> : hipblasDotBatched<T, true>)
                  : (CONJ ? hipblasDotcBatched<T, false> : hipblasDotBatched<T, false>);

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<T> d_hipblas_result_0(std::max(batch_count, 1));
        host_vector<T>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        ASSERT_HIP_SUCCESS(hipMemcpy(d_hipblas_result_0,
                                     h_hipblas_result_0,
                                     sizeof(T) * std::max(1, batch_count),
                                     hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasDotBatchedFn(
            handle, N, nullptr, incx, nullptr, incy, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<T> cpu_0(batch_count);
            host_vector<T> gpu_0(batch_count);
            ASSERT_HIP_SUCCESS(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(T) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<T>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return;
    }

    int    abs_incx = incx >= 0 ? incx : -incx;
    int    abs_incy = incy >= 0 ? incy : -incy;
    size_t sizeX    = size_t(N) * abs_incx;
    size_t sizeY    = size_t(N) * abs_incy;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_vector<T>       h_cpu_result(batch_count);
    host_vector<T>       h_hipblas_result1(batch_count);
    host_vector<T>       h_hipblas_result2(batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_hipblas_result(batch_count);
    ASSERT_HIP_SUCCESS(dx.memcheck());
    ASSERT_HIP_SUCCESS(dy.memcheck());

    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false);
    ASSERT_HIP_SUCCESS(dx.transfer_from(hx));
    ASSERT_HIP_SUCCESS(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        // hipblasDot accept both dev/host pointer for the scalar
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS((hipblasDotBatchedFn)(handle,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     incy,
                                                     batch_count,
                                                     d_hipblas_result));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS((hipblasDotBatchedFn)(handle,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     incy,
                                                     batch_count,
                                                     h_hipblas_result1));

        ASSERT_HIP_SUCCESS(hipMemcpy(
            h_hipblas_result2, d_hipblas_result, sizeof(T) * batch_count, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx[b], incx, hy[b], incy, &(h_cpu_result[b]));
        }

        if(arg.unit_check)
        {
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_hipblas_result1);
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_hipblas_result2);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, batch_count, 1, h_cpu_result, h_hipblas_result1);
            hipblas_error_device
                = norm_check_general<T>('F', 1, batch_count, 1, h_cpu_result, h_hipblas_result2);
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

            ASSERT_HIPBLAS_SUCCESS((hipblasDotBatchedFn)(handle,
                                                         N,
                                                         dx.ptr_on_device(),
                                                         incx,
                                                         dy.ptr_on_device(),
                                                         incy,
                                                         batch_count,
                                                         d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotBatchedModel{}.log_args<T>(std::cout,
                                             arg,
                                             gpu_time_used,
                                             dot_gflop_count<CONJ, T>(N),
                                             dot_gbyte_count<T>(N),
                                             hipblas_error_host,
                                             hipblas_error_device);
    }
}

template <typename T>
void testing_dotc_batched(const Arguments& arg)
{
    testing_dot_batched<T, true>(arg);
}
