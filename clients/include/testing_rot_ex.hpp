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

template <typename Tex, typename Tx = Tex, typename Tcs = Tx>
hipblasStatus_t testing_rot_ex_template(const Arguments& arg)
{
    using Ty            = Tx;
    bool FORTRAN        = arg.fortran;
    auto hipblasRotExFn = FORTRAN ? hipblasRotExFortran : hipblasRotEx;

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasRotExFn(handle,
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
                                           executionType));
        return HIPBLAS_STATUS_SUCCESS;
    }

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    int    abs_incx = incx >= 0 ? incx : -incx;
    int    abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * size_t(abs_incx);
    size_t size_y   = N * size_t(abs_incy);
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    device_vector<Tx>  dx(size_x);
    device_vector<Ty>  dy(size_y);
    device_vector<Tcs> dc(1);
    device_vector<Tcs> ds(1);

    // Initial Data on CPU
    host_vector<Tx>  hx_host(size_x);
    host_vector<Ty>  hy_host(size_y);
    host_vector<Tx>  hx_device(size_x);
    host_vector<Ty>  hy_device(size_y);
    host_vector<Tx>  hx_cpu(size_x);
    host_vector<Ty>  hy_cpu(size_y);
    host_vector<Tcs> hc(1);
    host_vector<Tcs> hs(1);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);

    hipblas_init_vector(hx_host, arg, N, abs_incx, 0, 1, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hy_host, arg, N, abs_incy, 0, 1, hipblas_client_never_set_nan, false);
    hipblas_init_vector(alpha, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hc, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);
    hipblas_init_vector(hs, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);

    // // cos and sin of alpha (in rads)
    // hc[0] = cos(alpha[0]);
    // hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    hx_cpu = hx_device = hx_host;
    hy_cpu = hy_device = hy_host;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx_host, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_host, sizeof(Ty) * size_y, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // HIPBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasRotExFn(
            handle, N, dx, xType, incx, dy, yType, incy, hc, hs, csType, executionType));

        CHECK_HIP_ERROR(hipMemcpy(hx_host, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_host, dy, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx_device, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy_device, sizeof(Ty) * size_y, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasRotExFn(
            handle, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, executionType));

        CHECK_HIP_ERROR(hipMemcpy(hx_device, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_device, dy, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));

        // CBLAS
        cblas_rot<Tx, Tcs, Tcs>(N, hx_cpu.data(), incx, hy_cpu.data(), incy, *hc, *hs);

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, abs_incx, hx_cpu, hx_host);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_host);
            unit_check_general<Tx>(1, N, abs_incx, hx_cpu, hx_device);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_host);
            hipblas_error_host += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<Tx>('F', 1, N, abs_incx, hx_cpu, hx_device);
            hipblas_error_device += norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device);
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

            CHECK_HIPBLAS_ERROR(hipblasRotExFn(
                handle, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<Tx>(std::cout,
                                                          arg,
                                                          gpu_time_used,
                                                          rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
                                                          rot_gbyte_count<Tx>(N),
                                                          hipblas_error_host,
                                                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_rot_ex(Arguments arg)
{
    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16B
       && csType == HIPBLAS_R_16B)
    {
        status = testing_rot_ex_template<float, hipblasBfloat16, hipblasBfloat16>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16F
            && csType == HIPBLAS_R_16F)
    {
        status = testing_rot_ex_template<float, hipblasHalf, hipblasHalf>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_ex_template<float>(arg);
    }
    else if(executionType == HIPBLAS_R_64F && xType == yType && xType == HIPBLAS_R_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_ex_template<double>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_ex_template<hipblasComplex, hipblasComplex, float>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_C_32F)
    {
        status = testing_rot_ex_template<hipblasComplex>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_ex_template<hipblasDoubleComplex, hipblasDoubleComplex, double>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_C_64F)
    {
        status = testing_rot_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
