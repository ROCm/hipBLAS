/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

template <typename T>
using hipblas_iamax_iamin_t
    = hipblasStatus_t (*)(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
hipblasStatus_t testing_iamax_iamin(const Arguments& argus, hipblas_iamax_iamin_t<T> func)
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

        status_1 = func(handle, N, dx, incx, &rocblas_result1);

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

            status_1 = func(handle, N, dx, incx, d_rocblas_result);

            CHECK_HIP_ERROR(
                hipMemcpy(&rocblas_result1, d_rocblas_result, sizeof(int), hipMemcpyDeviceToHost));
        }
        // host_pointer for rocblas_result2
        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

            status_2 = func(handle, N, dx, incx, &rocblas_result2);
        }

        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_2 == HIPBLAS_STATUS_SUCCESS)
           && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            /* =====================================================================
                        CPU BLAS
            =================================================================== */
            REFBLAS_FUNC(N, hx.data(), incx, &cpu_result);
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

template <typename T>
hipblasStatus_t testing_amax(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasIamaxFn = FORTRAN ? hipblasIamax<T, true> : hipblasIamax<T, false>;

    return testing_iamax_iamin<T, cblas_iamax<T>>(arg, hipblasIamaxFn);
}

template <typename T>
hipblasStatus_t testing_amin(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasIaminFn = FORTRAN ? hipblasIamin<T, true> : hipblasIamin<T, false>;

    return testing_iamax_iamin<T, cblas_iamin<T>>(arg, hipblasIamin<T>);
}
