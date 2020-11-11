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

template <typename T, typename U = T>
hipblasStatus_t testing_scal_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasScalBatchedFn
        = FORTRAN ? hipblasScalBatched<T, U, true> : hipblasScalBatched<T, U, false>;

    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int    sizeX = N * incx;
    U      alpha = argus.alpha;
    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx_array[batch_count];
    host_vector<T> hz_array[batch_count];

    device_batch_vector<T> bx_array(batch_count, sizeX);
    device_batch_vector<T> bz_array(batch_count, sizeX);

    device_vector<T*, 0, T> dx_array(batch_count);
    device_vector<T*, 0, T> dz_array(batch_count);

    int last = batch_count - 1;
    if(!dx_array || !dz_array || (!bx_array[last] && sizeX) || (!bz_array[last] && sizeX))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hx_array[b] = host_vector<T>(sizeX);
        hz_array[b] = host_vector<T>(sizeX);

        srand(1);
        hipblas_init<T>(hx_array[b], 1, N, incx);
        hz_array[b] = hx_array[b];

        CHECK_HIP_ERROR(
            hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * sizeX, hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasScalBatchedFn(handle, N, &alpha, dx_array, incx, batch_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(hx_array[b], bx_array[b], sizeof(T) * sizeX, hipMemcpyDeviceToHost));
    }

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<T, U>(N, alpha, hz_array[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, hz_array, hx_array);
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
