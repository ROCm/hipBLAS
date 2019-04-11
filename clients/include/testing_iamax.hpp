/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
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
#include <complex.h>

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_amax(Arguments argus)
{
    int N    = argus.N;
    int incx = argus.incx;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    T*   dx;
    int* d_rocblas_result;

    int cpu_result, rocblas_result1, rocblas_result2;
    int zero = 0;

    // check to prevent undefined memory allocation error
    if(N < 1 || incx <= 0)
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(int)));

        status_1 = hipblasIamax<T>(handle, N, dx, incx, &rocblas_result1);

        unit_check_general<int>(1, 1, 1, &zero, &rocblas_result1);
    }
    else
    {
        int sizeX = N * incx;

        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this
        // practice
        vector<T> hx(sizeX);

        // allocate memory on device
        CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(int)));

        // Initial Data on CPU
        srand(1);
        hipblas_init<T>(hx, 1, N, incx);

        // copy data from CPU to device, does not work for incx != 1
        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice));

        /* =====================================================================
                    HIP BLAS
        =================================================================== */
        // device_pointer for d_rocblas_result
        {

            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

            status_1 = hipblasIamax<T>(handle, N, dx, incx, d_rocblas_result);

            CHECK_HIP_ERROR(
                hipMemcpy(&rocblas_result1, d_rocblas_result, sizeof(int), hipMemcpyDeviceToHost));
        }
        // host_pointer for rocblas_result2
        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

            status_2 = hipblasIamax<T>(handle, N, dx, incx, &rocblas_result2);
        }

        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_2 == HIPBLAS_STATUS_SUCCESS)
           && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            /* =====================================================================
                        CPU BLAS
            =================================================================== */
            cblas_iamax<T>(N, hx.data(), incx, &cpu_result);
            // change to Fortran 1 based indexing as in BLAS standard, not cblas zero based indexing
            cpu_result += 1;

            unit_check_general<int>(1, 1, 1, &cpu_result, &rocblas_result1);
            unit_check_general<int>(1, 1, 1, &cpu_result, &rocblas_result2);

        } // end of if unit/norm check
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    hipblasDestroy(handle);

    if(status_1 != HIPBLAS_STATUS_SUCCESS)
    {
        return status_1;
    }
    else if(status_2 != HIPBLAS_STATUS_SUCCESS)
    {
        return status_2;
    }
    else if(status_3 != HIPBLAS_STATUS_SUCCESS)
    {
        return status_3;
    }
    else
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
}
