/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, bool CONJ = false>
hipblasStatus_t testing_dot(const Arguments& argus)
{
    bool FORTRAN      = argus.fortran;
    auto hipblasDotFn = FORTRAN ? (CONJ ? hipblasDotc<T, true> : hipblasDot<T, true>)
                                : (CONJ ? hipblasDotc<T, false> : hipblasDot<T, false>);

    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0)
    {
        status_1 = HIPBLAS_STATUS_INVALID_VALUE;
        return status_1;
    }
    else if(incx < 0)
    {
        status_1 = HIPBLAS_STATUS_INVALID_VALUE;
        return status_1;
    }
    else if(incy < 0)
    {
        status_1 = HIPBLAS_STATUS_INVALID_VALUE;
        return status_1;
    }

    int sizeX = N * incx;
    int sizeY = N * incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);

    T   cpu_result, rocblas_result;
    T * dx, *dy, *d_rocblas_result;
    int device_pointer = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T)));

    // Initial Data on CPU
    srand(1);
    hipblas_init_alternating_sign<T>(hx, 1, N, incx);
    hipblas_init<T>(hy, 1, N, incy);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * N * incy, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    /* =====================================================================
                CPU BLAS
    =================================================================== */
    // hipblasDot accept both dev/host pointer for the scalar
    if(device_pointer)
    {

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        status_2 = (hipblasDotFn)(handle, N, dx, incx, dy, incy, d_rocblas_result);
    }
    else
    {

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = (hipblasDotFn)(handle, N, dx, incx, dy, incy, &rocblas_result);
    }

    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
    {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        CHECK_HIP_ERROR(hipFree(d_rocblas_result));
        hipblasDestroy(handle);
        if(status_1 != HIPBLAS_STATUS_SUCCESS)
            return status_1;
        if(status_2 != HIPBLAS_STATUS_SUCCESS)
            return status_2;
    }

    if(device_pointer)
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(T), hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        if(argus.unit_check)
        {
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result);
        }

    } // end of if unit/norm check

    //  BLAS_1_RESULT_PRINT

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_dotc(const Arguments& argus)
{
    return testing_dot<T, true>(argus);
}
