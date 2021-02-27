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
hipblasStatus_t testing_scal_strided_batched_ex_template(Arguments argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasScalStridedBatchedExFn
        = FORTRAN ? hipblasScalStridedBatchedExFortran : hipblasScalStridedBatchedEx;

    int    N            = argus.N;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasStride stridex = N * incx * stride_scale;
    int           sizeX   = stridex * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    Ta alpha = argus.alpha;

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
    host_vector<Tx>   hx(sizeX);
    host_vector<Tx>   hz(sizeX);
    device_vector<Tx> dx(sizeX);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Tx>(hx, 1, N, incx, stridex, batch_count);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(Tx) * sizeX, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasScalStridedBatchedExFn(
        handle, N, &alpha, alphaType, dx, xType, incx, stridex, batch_count, executionType);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<Tx, Ta>(N, alpha, hz.data() + b * stridex, incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, incx, stridex, hz, hx);
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_scal_strided_batched_ex(Arguments argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_16F)
    {
        status = testing_scal_strided_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_strided_batched_ex_template<hipblasHalf, hipblasHalf, float>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_strided_batched_ex_template<float, hipblasHalf, float>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal_strided_batched_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_scal_strided_batched_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal_strided_batched_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status = testing_scal_strided_batched_ex_template<hipblasDoubleComplex>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal_strided_batched_ex_template<float, hipblasComplex, hipblasComplex>(
            argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status = testing_scal_strided_batched_ex_template<double,
                                                          hipblasDoubleComplex,
                                                          hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
