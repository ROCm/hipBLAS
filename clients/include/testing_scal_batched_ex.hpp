/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
hipblasStatus_t testing_scal_batched_ex_template(const Arguments& argus)
{
    bool FORTRAN                = argus.fortran;
    auto hipblasScalBatchedExFn = FORTRAN ? hipblasScalBatchedExFortran : hipblasScalBatchedEx;

    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;
    int unit_check  = argus.unit_check;
    int timing      = argus.timing;
    int norm_check  = argus.norm_check;

    Ta h_alpha = argus.get_alpha<Ta>();

    hipblasLocalHandle handle(argus);

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasScalBatchedExFn(
            handle, N, nullptr, alphaType, nullptr, xType, incx, batch_count, executionType));
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx_host(N, incx, batch_count);
    host_batch_vector<Tx> hx_device(N, incx, batch_count);
    host_batch_vector<Tx> hx_cpu(N, incx, batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_vector<Ta>       d_alpha(1);

    CHECK_HIP_ERROR(dx.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init(hx_host, true);

    hx_device.copy_from(hx_host);
    hx_cpu.copy_from(hx_host);

    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasScalBatchedExFn(handle,
                                                   N,
                                                   &h_alpha,
                                                   alphaType,
                                                   dx.ptr_on_device(),
                                                   xType,
                                                   incx,
                                                   batch_count,
                                                   executionType));

        CHECK_HIP_ERROR(hx_host.transfer_from(dx));
        CHECK_HIP_ERROR(dx.transfer_from(hx_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasScalBatchedExFn(handle,
                                                   N,
                                                   d_alpha,
                                                   alphaType,
                                                   dx.ptr_on_device(),
                                                   xType,
                                                   incx,
                                                   batch_count,
                                                   executionType));

        CHECK_HIP_ERROR(hx_device.transfer_from(dx));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<Tx, Ta>(N, h_alpha, hx_cpu[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, incx, hx_cpu, hx_host);
            unit_check_general<Tx>(1, N, batch_count, incx, hx_cpu, hx_device);
        }

        if(norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tx>('F', 1, N, incx, hx_cpu, hx_host, batch_count);
            hipblas_error_host
                = norm_check_general<Tx>('F', 1, N, incx, hx_cpu, hx_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasScalBatchedExFn(handle,
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

        ArgumentModel<e_N, e_alpha, e_incx, e_batch_count>{}.log_args<Tx>(
            std::cout,
            argus,
            gpu_time_used,
            scal_gflop_count<Tx, Ta>(N),
            scal_gbyte_count<Tx>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_scal_batched_ex(const Arguments& argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_16F)
    {
        status = testing_scal_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_batched_ex_template<hipblasHalf, hipblasHalf, float>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_batched_ex_template<float, hipblasHalf, float>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_batched_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_scal_batched_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal_batched_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status = testing_scal_batched_ex_template<hipblasDoubleComplex>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal_batched_ex_template<float, hipblasComplex, hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status
            = testing_scal_batched_ex_template<double, hipblasDoubleComplex, hipblasDoubleComplex>(
                argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
