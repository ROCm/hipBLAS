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

using hipblasScalBatchedExModel
    = ArgumentModel<e_a_type, e_b_type, e_compute_type, e_N, e_alpha, e_incx, e_batch_count>;

inline void testname_scal_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasScalBatchedExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_batched_ex(const Arguments& arg)
{
    bool FORTRAN                = arg.fortran;
    auto hipblasScalBatchedExFn = FORTRAN ? hipblasScalBatchedExFortran : hipblasScalBatchedEx;

    int N           = arg.N;
    int incx        = arg.incx;
    int batch_count = arg.batch_count;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;
    int norm_check = arg.norm_check;

    Ta h_alpha = arg.get_alpha<Ta>();

    hipblasLocalHandle handle(arg);

    hipblasDatatype_t alphaType     = arg.a_type;
    hipblasDatatype_t xType         = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        ASSERT_HIPBLAS_SUCCESS(hipblasScalBatchedExFn(
            handle, N, nullptr, alphaType, nullptr, xType, incx, batch_count, executionType));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx_host(N, incx, batch_count);
    host_batch_vector<Tx> hx_device(N, incx, batch_count);
    host_batch_vector<Tx> hx_cpu(N, incx, batch_count);

    host_batch_vector<Tex> hx_cpu_ex(N, incx, batch_count);
    host_batch_vector<Tex> hx_host_ex(N, incx, batch_count);
    host_batch_vector<Tex> hx_device_ex(N, incx, batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_vector<Ta>       d_alpha(1);

    ASSERT_HIP_SUCCESS(dx.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx_host, arg, hipblas_client_alpha_sets_nan, true);

    hx_device.copy_from(hx_host);
    hx_cpu.copy_from(hx_host);

    ASSERT_HIP_SUCCESS(dx.transfer_from(hx_host));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasScalBatchedExFn(handle,
                                                      N,
                                                      &h_alpha,
                                                      alphaType,
                                                      dx.ptr_on_device(),
                                                      xType,
                                                      incx,
                                                      batch_count,
                                                      executionType));

        ASSERT_HIP_SUCCESS(hx_host.transfer_from(dx));
        ASSERT_HIP_SUCCESS(dx.transfer_from(hx_device));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasScalBatchedExFn(handle,
                                                      N,
                                                      d_alpha,
                                                      alphaType,
                                                      dx.ptr_on_device(),
                                                      xType,
                                                      incx,
                                                      batch_count,
                                                      executionType));

        ASSERT_HIP_SUCCESS(hx_device.transfer_from(dx));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<Tx, Ta>(N, h_alpha, hx_cpu[b], incx);
        }

        for(size_t b = 0; b < batch_count; b++)
        {
            for(size_t i = 0; i < N; i++)
            {
                hx_host_ex[b][i * incx]   = (Tex)hx_host[b][i * incx];
                hx_device_ex[b][i * incx] = (Tex)hx_device[b][i * incx];
                hx_cpu_ex[b][i * incx]    = (Tex)hx_cpu[b][i * incx];
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tex>(1, N, batch_count, incx, hx_cpu_ex, hx_host_ex);
            unit_check_general<Tex>(1, N, batch_count, incx, hx_cpu_ex, hx_device_ex);
        }

        if(norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tex>('F', 1, N, incx, hx_cpu_ex, hx_host_ex, batch_count);
            hipblas_error_host
                = norm_check_general<Tex>('F', 1, N, incx, hx_cpu_ex, hx_device_ex, batch_count);
        }
    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(hipblasScalBatchedExFn(handle,
                                                          N,
                                                          d_alpha,
                                                          alphaType,
                                                          dx.ptr_on_device(),
                                                          xType,
                                                          incx,
                                                          batch_count,
                                                          executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalBatchedExModel{}.log_args<Tx>(std::cout,
                                                 arg,
                                                 gpu_time_used,
                                                 scal_gflop_count<Tx, Ta>(N),
                                                 scal_gbyte_count<Tx>(N),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }
}
