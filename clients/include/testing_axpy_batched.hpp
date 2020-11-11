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
hipblasStatus_t testing_axpy_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasAxpyBatchedFn
        = FORTRAN ? hipblasAxpyBatched<T, true> : hipblasAxpyBatched<T, false>;

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

    hipblasStatus_t status   = HIPBLAS_STATUS_SUCCESS;
    int             abs_incx = incx < 0 ? -incx : incx;
    int             abs_incy = incy < 0 ? -incy : incy;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || !incx || !incy || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int sizeX = N * abs_incx;
    int sizeY = N * abs_incy;
    T   alpha = argus.alpha;

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx_array[batch_count];
    host_vector<T> hy_array[batch_count];
    host_vector<T> hx_cpu_array[batch_count];
    host_vector<T> hy_cpu_array[batch_count];

    device_batch_vector<T> bx_array(batch_count, sizeX);
    device_batch_vector<T> by_array(batch_count, sizeY);

    device_vector<T*, 0, T> dx_array(batch_count);
    device_vector<T*, 0, T> dy_array(batch_count);

    int last = batch_count - 1;
    if(!dx_array || !dy_array || (!bx_array[last] && sizeX) || (!by_array[last] && sizeY))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hx_array[b]     = host_vector<T>(sizeX);
        hy_array[b]     = host_vector<T>(sizeY);
        hx_cpu_array[b] = host_vector<T>(sizeX);
        hy_cpu_array[b] = host_vector<T>(sizeY);

        srand(1);
        hipblas_init(hx_array[b], 1, N, abs_incx);
        hipblas_init(hy_array[b], 1, N, abs_incy);

        hx_cpu_array[b] = hx_array[b];
        hy_cpu_array[b] = hy_array[b];

        CHECK_HIP_ERROR(
            hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * sizeX, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(by_array[b], hy_array[b], sizeof(T) * sizeY, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dx_array, bx_array, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_array, by_array, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasAxpyBatchedFn(handle, N, &alpha, dx_array, incx, dy_array, incy, batch_count);
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
        CHECK_HIP_ERROR(
            hipMemcpy(hy_array[b], by_array[b], sizeof(T) * sizeY, hipMemcpyDeviceToHost));
    }

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy<T>(N, alpha, hx_cpu_array[b], incx, hy_cpu_array[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, hx_cpu_array, hx_array);
            unit_check_general<T>(1, N, batch_count, abs_incy, hy_cpu_array, hy_array);
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
