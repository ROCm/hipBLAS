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

// Tolerance of 100 fails for complex,
// TODO: something better than arbitrary tolerance.
template <typename T>
constexpr double nrm2_tolerance_multiplier = 100;
template <>
constexpr double nrm2_tolerance_multiplier<hipblasComplex> = 110;
template <>
constexpr double nrm2_tolerance_multiplier<hipblasDoubleComplex> = 110;

template <typename T1, typename T2>
hipblasStatus_t testing_nrm2(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasNrm2Fn = FORTRAN ? hipblasNrm2<T1, T2, true> : hipblasNrm2<T1, T2, false>;

    int N    = argus.N;
    int incx = argus.incx;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    // check to prevent undefined memory allocation error
    if(N < 0 || incx < 0)
    {
        status_1 = HIPBLAS_STATUS_INVALID_VALUE;
        return status_1;
    }

    int sizeX = N * incx;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T1> hx(sizeX);

    T1* dx;
    T2* d_rocblas_result;
    T2  cpu_result, rocblas_result_1, rocblas_result_2;

    int device_pointer = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T1)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T2)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<T1>(hx, 1, N, incx);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T1) * N * incx, hipMemcpyHostToDevice));

    // hipblasNrm2 accept both dev/host pointer for the scalar

    status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

    status_2 = hipblasNrm2Fn(handle, N, dx, incx, d_rocblas_result);

    status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

    status_4 = hipblasNrm2Fn(handle, N, dx, incx, &rocblas_result_1);

    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS)
       || (status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
    {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(d_rocblas_result));
        hipblasDestroy(handle);
        if(status_1 != HIPBLAS_STATUS_SUCCESS)
            return status_1;
        if(status_2 != HIPBLAS_STATUS_SUCCESS)
            return status_2;
        if(status_3 != HIPBLAS_STATUS_SUCCESS)
            return status_3;
        if(status_4 != HIPBLAS_STATUS_SUCCESS)
            return status_4;
    }

    CHECK_HIP_ERROR(
        hipMemcpy(&rocblas_result_2, d_rocblas_result, sizeof(T2), hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {

        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        cblas_nrm2<T1, T2>(N, hx.data(), incx, &cpu_result);

        if(argus.unit_check)
        {
            T2 tolerance = nrm2_tolerance_multiplier<T1>;
            unit_check_nrm2<T2>(cpu_result, rocblas_result_1, tolerance);
            unit_check_nrm2<T2>(cpu_result, rocblas_result_2, tolerance);
        }

    } // end of if unit/norm check

    //  BLAS_1_RESULT_PRINT

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
