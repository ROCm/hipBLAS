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

template <typename T, bool CONJ = false>
hipblasStatus_t testing_dot_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDotBatchedFn
        = FORTRAN ? (CONJ ? hipblasDotcBatched<T, true> : hipblasDotBatched<T, true>)
                  : (CONJ ? hipblasDotcBatched<T, false> : hipblasDotBatched<T, false>);

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || incy < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int sizeX = N * incx;
    int sizeY = N * incy;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx_array[batch_count];
    host_vector<T> hy_array[batch_count];
    host_vector<T> h_cpu_result(batch_count);
    host_vector<T> h_hipblas_result1(batch_count);
    host_vector<T> h_hipblas_result2(batch_count);

    device_batch_vector<T> bx_array(batch_count, sizeX);
    device_batch_vector<T> by_array(batch_count, sizeY);

    device_vector<T*, 0, T> dx_array(batch_count);
    device_vector<T*, 0, T> dy_array(batch_count);
    device_vector<T>        d_hipblas_result(batch_count);

    int last = batch_count - 1;
    if(!dx_array || !dy_array || !d_hipblas_result || (!bx_array[last] && sizeX)
       || (!by_array[last] && sizeY))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hx_array[b] = host_vector<T>(sizeX);
        hy_array[b] = host_vector<T>(sizeY);

        srand(1);
        hipblas_init_alternating_sign<T>(hx_array[b], 1, N, incx);
        hipblas_init<T>(hy_array[b], 1, N, incy);

        CHECK_HIP_ERROR(
            hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * sizeX, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(by_array[b], hy_array[b], sizeof(T) * sizeY, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_array, by_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    // hipblasDot accept both dev/host pointer for the scalar
    status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    status_2 = (hipblasDotBatchedFn)(
        handle, N, dx_array, incx, dy_array, incy, batch_count, d_hipblas_result);

    status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    status_4 = (hipblasDotBatchedFn)(
        handle, N, dx_array, incx, dy_array, incy, batch_count, h_hipblas_result1);

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

    CHECK_HIP_ERROR(hipMemcpy(
        h_hipblas_result2, d_hipblas_result, sizeof(T) * batch_count, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<T>
                  : cblas_dot<T>)(N, hx_array[b], incx, hy_array[b], incy, &(h_cpu_result[b]));
        }

        if(argus.unit_check)
        {
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_hipblas_result1);
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_hipblas_result2);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, batch_count, 1, h_cpu_result, h_hipblas_result1);
            hipblas_error_device
                = norm_check_general<T>('F', 1, batch_count, 1, h_cpu_result, h_hipblas_result2);
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        hipStream_t stream;
        status_1 = hipblasGetStream(handle, &stream);
        status_2 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        if(status_1 != HIPBLAS_STATUS_SUCCESS || status_2 != HIPBLAS_STATUS_SUCCESS)
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

            status_1 = (hipblasDotBatchedFn)(
                handle, N, dx_array, incx, dy_array, incy, batch_count, d_hipblas_result);

            if(status_1 != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status_1;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.log_args<T>(std::cout,
                                                                        argus,
                                                                        gpu_time_used,
                                                                        dot_gflop_count<CONJ, T>(N),
                                                                        dot_gbyte_count<T>(N),
                                                                        hipblas_error_host,
                                                                        hipblas_error_device);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_dotc_batched(const Arguments& argus)
{
    return testing_dot_batched<T, true>(argus);
}
