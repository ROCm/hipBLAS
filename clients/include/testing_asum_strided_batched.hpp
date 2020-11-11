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

template <typename T1, typename T2>
hipblasStatus_t testing_asum_strided_batched(const Arguments& argus)
{
    int N            = argus.N;
    int incx         = argus.incx;
    int stride_scale = argus.stride_scale;
    int batch_count  = argus.batch_count;

    int stridex = N * incx * stride_scale;
    int sizeX   = stridex * batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    // check to prevent undefined memory allocation error
    if(N < 0 || incx < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        // return early so we don't get invalid_value from rocblas because of bad result pointer
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T1> hx(sizeX);
    host_vector<T2> cpu_result(batch_count);
    host_vector<T2> rocblas_result1(batch_count);
    host_vector<T2> rocblas_result2(batch_count);

    device_vector<T1> dx(sizeX);
    device_vector<T2> d_rocblas_result(batch_count);

    int device_pointer = 1;
    int host_pointer   = 1;

    // Initial Data on CPU
    srand(1);
    hipblas_init<T1>(hx, 1, N, incx, stridex, batch_count);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T1) * sizeX, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    // hipblasAsum accept both dev/host pointer for the scalar

    if(device_pointer)
    {
        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_2 = hipblasAsumStridedBatched<T1, T2>(
            handle, N, dx, incx, stridex, batch_count, d_rocblas_result);
    }

    if(host_pointer)
    {
        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        status_4 = hipblasAsumStridedBatched<T1, T2>(
            handle, N, dx, incx, stridex, batch_count, rocblas_result1);
    }

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

    if(device_pointer)
        CHECK_HIP_ERROR(hipMemcpy(
            rocblas_result2, d_rocblas_result, sizeof(T2) * batch_count, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_asum<T1, T2>(N, hx.data() + b * stridex, incx, &cpu_result[b]);
        }

        if(argus.unit_check)
        {
            unit_check_general<T2>(1, batch_count, 1, cpu_result, rocblas_result1);
            unit_check_general<T2>(1, batch_count, 1, cpu_result, rocblas_result2);
        }

    } // end of if unit/norm check

    //  BLAS_1_RESULT_PRINT
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
