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

using hipblasRotBatchedExModel = ArgumentModel<e_N, e_incx, e_incy, e_batch_count>;

inline void testname_rot_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasRotBatchedExModel{}.test_name(arg, name);
}

template <typename Tex, typename Tx = Tex, typename Tcs = Tx>
inline hipblasStatus_t testing_rot_batched_ex_template(const Arguments& arg)
{
    using Ty                   = Tx;
    bool FORTRAN               = arg.fortran;
    auto hipblasRotBatchedExFn = FORTRAN ? hipblasRotBatchedExFortran : hipblasRotBatchedEx;

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasRotBatchedExFn(handle,
                                                  N,
                                                  nullptr,
                                                  xType,
                                                  incx,
                                                  nullptr,
                                                  yType,
                                                  incy,
                                                  nullptr,
                                                  nullptr,
                                                  csType,
                                                  batch_count,
                                                  executionType));

        return HIPBLAS_STATUS_SUCCESS;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    host_batch_vector<Tx> hx_host(N, incx, batch_count);
    host_batch_vector<Ty> hy_host(N, incy, batch_count);
    host_batch_vector<Tx> hx_device(N, incx, batch_count);
    host_batch_vector<Ty> hy_device(N, incy, batch_count);
    host_batch_vector<Tx> hx_cpu(N, incx, batch_count);
    host_batch_vector<Ty> hy_cpu(N, incy, batch_count);
    host_vector<Tcs>      hc(1);
    host_vector<Tcs>      hs(1);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tcs>      dc(1);
    device_vector<Tcs>      ds(1);

    hipblas_init_vector(hx_host, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hc, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hs, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);

    hx_device.copy_from(hx_host);
    hx_cpu.copy_from(hx_host);
    hy_device.copy_from(hy_host);
    hy_cpu.copy_from(hy_host);

    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));

    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // HIPBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasRotBatchedExFn(handle,
                                                  N,
                                                  dx.ptr_on_device(),
                                                  xType,
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  yType,
                                                  incy,
                                                  hc,
                                                  hs,
                                                  csType,
                                                  batch_count,
                                                  executionType));

        CHECK_HIP_ERROR(hx_host.transfer_from(dx));
        CHECK_HIP_ERROR(hy_host.transfer_from(dy));
        CHECK_HIP_ERROR(dx.transfer_from(hx_device));
        CHECK_HIP_ERROR(dy.transfer_from(hy_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasRotBatchedExFn(handle,
                                                  N,
                                                  dx.ptr_on_device(),
                                                  xType,
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  yType,
                                                  incy,
                                                  dc,
                                                  ds,
                                                  csType,
                                                  batch_count,
                                                  executionType));

        CHECK_HIP_ERROR(hx_device.transfer_from(dx));
        CHECK_HIP_ERROR(hy_device.transfer_from(dy));

        // CBLAS
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rot<Tx, Tcs, Tcs>(N, hx_cpu[b], incx, hy_cpu[b], incy, *hc, *hs);
        }

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, abs_incx, hx_cpu, hx_host);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_host);
            unit_check_general<Tx>(1, N, batch_count, abs_incx, hx_cpu, hx_device);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_host, batch_count);
            hipblas_error_host
                += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_device, batch_count);
            hipblas_error_device
                += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasRotBatchedExFn(handle,
                                                      N,
                                                      dx.ptr_on_device(),
                                                      xType,
                                                      incx,
                                                      dy.ptr_on_device(),
                                                      yType,
                                                      incy,
                                                      dc,
                                                      ds,
                                                      csType,
                                                      batch_count,
                                                      executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotBatchedExModel{}.log_args<Tx>(std::cout,
                                                arg,
                                                gpu_time_used,
                                                rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
                                                rot_gbyte_count<Tx>(N),
                                                hipblas_error_host,
                                                hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

inline hipblasStatus_t testing_rot_batched_ex(Arguments arg)
{
    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16B
       && csType == HIPBLAS_R_16B)
    {
        status = testing_rot_batched_ex_template<float, hipblasBfloat16, hipblasBfloat16>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16F
            && csType == HIPBLAS_R_16F)
    {
        status = testing_rot_batched_ex_template<float, hipblasHalf, hipblasHalf>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_batched_ex_template<float>(arg);
    }
    else if(executionType == HIPBLAS_R_64F && xType == yType && xType == HIPBLAS_R_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_batched_ex_template<double>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_batched_ex_template<hipblasComplex, hipblasComplex, float>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_C_32F)
    {
        status = testing_rot_batched_ex_template<hipblasComplex>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_R_64F)
    {
        status
            = testing_rot_batched_ex_template<hipblasDoubleComplex, hipblasDoubleComplex, double>(
                arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_C_64F)
    {
        status = testing_rot_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
