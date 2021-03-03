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

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int sizeX = N * incx;
    Ta  alpha = argus.get_alpha<Ta>();

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx[batch_count];
    host_vector<Tx> hz[batch_count];

    device_batch_vector<Tx> bx(batch_count, sizeX);
    device_batch_vector<Tx> bz(batch_count, sizeX);

    device_vector<Tx*, 0, Tx> dx(batch_count);
    device_vector<Tx*, 0, Tx> dz(batch_count);

    int last = batch_count - 1;
    if(!dx || !dz || (!bx[last] && sizeX) || (!bz[last] && sizeX))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hx[b] = host_vector<Tx>(sizeX);
        hz[b] = host_vector<Tx>(sizeX);

        srand(1);
        hipblas_init<Tx>(hx[b], 1, N, incx);
        hz[b] = hx[b];

        CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dx, bx, batch_count * sizeof(Tx*), hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasScalBatchedExFn(
        handle, N, &alpha, alphaType, dx, xType, incx, batch_count, executionType);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(hipMemcpy(hx[b], bx[b], sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));
    }

    if(unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<Tx, Ta>(N, alpha, hz[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, incx, hz, hx);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        status = hipblasGetStream(handle, &stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            status = hipblasScalBatchedExFn(
                handle, N, &alpha, alphaType, dx, xType, incx, batch_count, executionType);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_batch_count>{}.log_args<Tx>(
            std::cout,
            argus,
            gpu_time_used,
            scal_gflop_count<Tx, Ta>(N),
            scal_gbyte_count<Tx>(N),
            hipblas_error);
    }
    hipblasDestroy(handle);
    return status;
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
