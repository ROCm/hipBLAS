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

using hipblasScalExModel = ArgumentModel<e_N, e_alpha, e_incx>;

inline void testname_scal_ex(const Arguments& arg, std::string& name)
{
    hipblasScalExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
inline hipblasStatus_t testing_scal_ex_template(const Arguments& arg)
{
    bool FORTRAN         = arg.fortran;
    auto hipblasScalExFn = FORTRAN ? hipblasScalExFortran : hipblasScalEx;

    int N    = arg.N;
    int incx = arg.incx;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;
    int norm_check = arg.norm_check;

    size_t sizeX   = size_t(N) * incx;
    Ta     h_alpha = arg.get_alpha<Ta>();

    hipblasLocalHandle handle(arg);

    hipblasDatatype_t alphaType     = arg.a_type;
    hipblasDatatype_t xType         = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0)
    {
        CHECK_HIPBLAS_ERROR(
            hipblasScalExFn(handle, N, nullptr, alphaType, nullptr, xType, incx, executionType));
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx_host(sizeX);
    host_vector<Tx> hx_device(sizeX);
    host_vector<Tx> hx_cpu(sizeX);

    device_vector<Tx> dx(sizeX);
    device_vector<Ta> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx_host, arg, N, incx, 0, 1, hipblas_client_alpha_sets_nan, true);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hx_device = hx_cpu = hx_host;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx_host, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            hipblasScalExFn(handle, N, &h_alpha, alphaType, dx, xType, incx, executionType));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_host, dx, sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx_device, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            hipblasScalExFn(handle, N, d_alpha, alphaType, dx, xType, incx, executionType));

        CHECK_HIP_ERROR(hipMemcpy(hx_device, dx, sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_scal<Tx, Ta>(N, h_alpha, hx_cpu, incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tx>(1, N, incx, hx_cpu, hx_host);
            unit_check_general<Tx>(1, N, incx, hx_cpu, hx_device);
        }

        if(norm_check)
        {
            hipblas_error_host = norm_check_general<Tx>('F', 1, N, incx, hx_cpu, hx_host);
            hipblas_error_host = norm_check_general<Tx>('F', 1, N, incx, hx_cpu, hx_device);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasScalExFn(handle, N, d_alpha, alphaType, dx, xType, incx, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalExModel{}.log_args<Tx>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          scal_gflop_count<Tx, Ta>(N),
                                          scal_gbyte_count<Tx>(N),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

inline hipblasStatus_t testing_scal_ex(const Arguments& arg)
{
    hipblasDatatype_t alphaType     = arg.a_type;
    hipblasDatatype_t xType         = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIP_R_16F && xType == HIP_R_16F && executionType == HIP_R_16F)
    {
        status = testing_scal_ex_template<hipblasHalf>(arg);
    }
    else if(alphaType == HIP_R_16F && xType == HIP_R_16F && executionType == HIP_R_32F)
    {
        status = testing_scal_ex_template<hipblasHalf, hipblasHalf, float>(arg);
    }
    else if(alphaType == HIP_R_32F && xType == HIP_R_16F && executionType == HIP_R_32F)
    {
        status = testing_scal_ex_template<float, hipblasHalf, float>(arg);
    }
    else if(alphaType == HIP_R_32F && xType == HIP_R_32F && executionType == HIP_R_32F)
    {
        status = testing_scal_ex_template<float>(arg);
    }
    else if(alphaType == HIP_R_64F && xType == HIP_R_64F && executionType == HIP_R_64F)
    {
        status = testing_scal_ex_template<double>(arg);
    }
    else if(alphaType == HIP_C_32F && xType == HIP_C_32F && executionType == HIP_C_32F)
    {
        status = testing_scal_ex_template<hipblasComplex>(arg);
    }
    else if(alphaType == HIP_C_64F && xType == HIP_C_64F && executionType == HIP_C_64F)
    {
        status = testing_scal_ex_template<hipblasDoubleComplex>(arg);
    }
    else if(alphaType == HIP_R_32F && xType == HIP_C_32F && executionType == HIP_C_32F)
    {
        status = testing_scal_ex_template<float, hipblasComplex, hipblasComplex>(arg);
    }
    else if(alphaType == HIP_R_64F && xType == HIP_C_64F && executionType == HIP_C_64F)
    {
        status = testing_scal_ex_template<double, hipblasDoubleComplex, hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
