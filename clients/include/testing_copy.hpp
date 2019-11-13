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
hipblasStatus_t testing_copy(Arguments argus)
{
    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incx < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    int sizeX = N * incx;
    int sizeY = N * incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);
    vector<T> hx_cpu(sizeX);
    vector<T> hy_cpu(sizeY);
    T*        dx;
    T*        dy;

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;

    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);
    hipblas_init<T>(hy, 1, N, incy);

    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasCopy<T>(handle, N, dx, incx, dy, incy);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_copy<T>(N, hx_cpu.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx_cpu.data(), hx.data());
            unit_check_general<T>(1, N, incy, hy_cpu.data(), hy.data());
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
