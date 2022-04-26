/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename Ta, typename Tx = Ta, typename Ty = Tx>
hipblasStatus_t testing_axpy_ex_template(const Arguments& argus)
{
    bool FORTRAN         = argus.fortran;
    auto hipblasAxpyExFn = FORTRAN ? hipblasAxpyExFortran : hipblasAxpyEx;

    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    hipblasLocalHandle handle(argus);

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(handle,
                                            N,
                                            nullptr,
                                            alphaType,
                                            nullptr,
                                            xType,
                                            incx,
                                            nullptr,
                                            yType,
                                            incy,
                                            executionType));
        return HIPBLAS_STATUS_SUCCESS;
    }

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    size_t sizeX = size_t(N) * abs_incx;
    size_t sizeY = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    Ta h_alpha = argus.get_alpha<Ta>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy_host(sizeY);
    host_vector<Tx> hy_device(sizeX);
    host_vector<Ty> hy_cpu(sizeY);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);
    device_vector<Ta> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, argus, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, argus, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);

    hy_device = hy_host;
    hy_cpu    = hy_host;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_host, sizeof(Ty) * sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
        handle, N, &h_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hy_host, dy, sizeof(Ty) * sizeY, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_device, sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
        handle, N, d_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));
    CHECK_HIP_ERROR(hipMemcpy(hy_device, dy, sizeof(Ty) * sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_axpy(N, h_alpha, hx.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_host);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host   = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device);
        }

    } // end of if unit check

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

            CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
                handle, N, d_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<Ta>(std::cout,
                                                          argus,
                                                          gpu_time_used,
                                                          axpy_gflop_count<Ta>(N),
                                                          axpy_gbyte_count<Ta>(N),
                                                          hipblas_error_host,
                                                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_axpy_ex(Arguments argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_axpy_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_ex_template<float, hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_axpy_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_axpy_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_axpy_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_axpy_ex_template<hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
