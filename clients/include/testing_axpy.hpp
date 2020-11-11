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

template <typename T>
hipblasStatus_t testing_axpy(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasAxpyFn = FORTRAN ? hipblasAxpy<T, true> : hipblasAxpy<T, false>;

    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || !incx || !incy)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int sizeX = N * abs_incx;
    int sizeY = N * abs_incy;
    T   alpha = argus.alpha;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeX);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, abs_incx);
    hipblas_init<T>(hy, 1, N, abs_incy);

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasAxpyFn(handle, N, &alpha, dx, incx, dy, incy);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
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
        cblas_axpy<T>(N, alpha, hx_cpu.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, hx_cpu.data(), hx.data());
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy.data());
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
