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

template <typename T>
hipblasStatus_t testing_swap(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasSwapFn = FORTRAN ? hipblasSwap<T, true> : hipblasSwap<T, false>;

    int N          = argus.N;
    int incx       = argus.incx;
    int incy       = argus.incy;
    int unit_check = argus.unit_check;
    int timing     = argus.timing;

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
    else if(incy < 0)
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

    // allocate memory on device
    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);
    int              device_pointer = 1;

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);
    hipblas_init<T>(hy, 1, N, incy);
    hx_cpu = hx;
    hy_cpu = hy;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasSwapFn(handle, N, dx, incx, dy, incy);

    if((status != HIPBLAS_STATUS_SUCCESS))
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

    if(unit_check)
    {

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_swap<T>(N, hx.data(), incx, hy.data(), incy);

        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx_cpu.data(), hx.data());
        }

    } // end of if unit/norm check

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

            status = hipblasSwapFn(handle, N, dx, incx, dy, incy);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<T>(std::cout,
                                                         argus,
                                                         gpu_time_used,
                                                         swap_gflop_count<T>(N),
                                                         swap_gbyte_count<T>(N),
                                                         hipblas_error);
    }

    hipblasDestroy(handle);
    return status;
}
