/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
hipblasStatus_t testing_scal_ex_template(const Arguments& argus)
{
    bool FORTRAN         = argus.fortran;
    auto hipblasScalExFn = FORTRAN ? hipblasScalExFortran : hipblasScalEx;

    int N          = argus.N;
    int incx       = argus.incx;
    int unit_check = argus.unit_check;
    int timing     = argus.timing;
    int norm_check = argus.norm_check;

    size_t sizeX   = size_t(N) * incx;
    Ta     h_alpha = argus.get_alpha<Ta>();

    hipblasLocalHandle handle(argus);

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

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
    hipblas_init_vector(hx_host, argus, N, incx, 0, 1, hipblas_client_alpha_sets_nan, true);

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

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasScalExFn(handle, N, d_alpha, alphaType, dx, xType, incx, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx>{}.log_args<Tx>(std::cout,
                                                           argus,
                                                           gpu_time_used,
                                                           scal_gflop_count<Tx, Ta>(N),
                                                           scal_gbyte_count<Tx>(N),
                                                           hipblas_error_host,
                                                           hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_scal_ex(const Arguments& argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_16F)
    {
        status = testing_scal_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_ex_template<hipblasHalf, hipblasHalf, float>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_ex_template<float, hipblasHalf, float>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_scal_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status = testing_scal_ex_template<hipblasDoubleComplex>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal_ex_template<float, hipblasComplex, hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status
            = testing_scal_ex_template<double, hipblasDoubleComplex, hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
