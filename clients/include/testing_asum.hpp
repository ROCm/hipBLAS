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
hipblasStatus_t testing_asum(const Arguments& argus)
{
    using Tr           = real_t<T>;
    bool FORTRAN       = argus.fortran;
    auto hipblasAsumFn = FORTRAN ? hipblasAsum<T, Tr, true> : hipblasAsum<T, Tr, false>;

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
    host_vector<T> hx(sizeX);

    device_vector<T>  dx(sizeX);
    device_vector<Tr> d_hipblas_result(1);
    Tr                cpu_result, hipblas_result_host, hipblas_result_device;

    double gpu_time_used, hipblas_error_host = 0, hipblas_error_device = 0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    // hipblasAsum accept both dev/host pointer for the scalar
    status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    status_2 = hipblasAsumFn(handle, N, dx, incx, d_hipblas_result);

    status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    status_2 = hipblasAsumFn(handle, N, dx, incx, &hipblas_result_host);

    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS)
       || (status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
    {
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
        hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        cblas_asum<T, Tr>(N, hx.data(), incx, &cpu_result);

        if(argus.unit_check)
        {
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host);
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('M', 1, 1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('M', 1, 1, 1, &cpu_result, &hipblas_result_device);
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        hipStream_t stream;
        status_1 = hipblasGetStream(handle, &stream);
        status_2 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_1 != HIPBLAS_STATUS_SUCCESS)
                return status_1;
            if(status_2 != HIPBLAS_STATUS_SUCCESS)
                return status_2;
        }

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            status_1 = hipblasAsumFn(handle, N, dx, incx, d_hipblas_result);

            if(status_1 != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status_1;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(std::cout,
                                                 argus,
                                                 gpu_time_used,
                                                 asum_gflop_count<T>(N),
                                                 asum_gbyte_count<T>(N),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
