/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_scal_ex_template(Arguments argus)
{
    bool FORTRAN         = argus.fortran;
    auto hipblasScalExFn = FORTRAN ? hipblasScalExFortran : hipblasScalEx;

    int N    = argus.N;
    int incx = argus.incx;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int sizeX = N * incx;
    T   alpha = argus.alpha;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t executionType = argus.compute_type;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T>   hx(sizeX);
    host_vector<T>   hz(sizeX);
    device_vector<T> dx(sizeX);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasScalExFn(handle, N, &alpha, dx, xType, incx, executionType);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * N * incx, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_scal<T, U>(N, alpha, hz.data(), incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, incx, hz.data(), hx.data());
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_scal_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal<hipblasHalf>(argus);
    }
    else if(xType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_scal<float>(argus);
    }
    else if(xType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_scal<double>(argus);
    }
    else if(xType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_scal<hipblasComplex>(argus);
    }
    else if(xType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status = testing_scal<hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
