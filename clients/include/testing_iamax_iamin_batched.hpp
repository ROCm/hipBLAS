/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

template <typename T>
using hipblas_iamax_iamin_batched_t = hipblasStatus_t (*)(
    hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
hipblasStatus_t testing_iamax_iamin_batched(const Arguments&                 argus,
                                            hipblas_iamax_iamin_batched_t<T> func)
{
    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // check to prevent undefined memory allocation error
    if(batch_count == 0)
    {
        // quick return success or invalid value
        device_vector<T*, 0, T> dx_array(5);
        device_vector<int>      d_rocblas_result(1);

        status_1 = func(handle, N, dx_array, incx, batch_count, d_rocblas_result);
    }
    else if(batch_count < 0)
    {
        status_1 = HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(N < 1 || incx <= 0)
    {
        // quick return success
        device_vector<T*, 0, T> dx_array(5);
        host_vector<int>        h_rocblas_result(batch_count);
        host_vector<int>        h_zeros(batch_count);
        for(int b = 0; b < batch_count; b++)
            h_zeros[b] = 0;

        status_1 = func(handle, N, dx_array, incx, batch_count, h_rocblas_result);
        unit_check_general<int>(1, 1, batch_count, h_zeros, h_rocblas_result);
    }

    else
    {
        host_vector<int>   cpu_result(batch_count);
        host_vector<int>   rocblas_result_host(batch_count);
        host_vector<int>   h_rocblas_result_device(batch_count);
        device_vector<int> rocblas_result_device(batch_count);

        int sizeX = N * incx;

        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this
        // practice
        host_vector<T> hx_array[batch_count];

        device_batch_vector<T> bx_array(batch_count, sizeX);

        device_vector<T*, 0, T> dx_array(batch_count);

        if(!dx_array || (!bx_array[batch_count - 1] && sizeX))
        {
            hipblasDestroy(handle);
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }

        // Initial Data on CPU
        srand(1);
        for(int b = 0; b < batch_count; b++)
        {
            hx_array[b] = host_vector<T>(sizeX);

            srand(1);
            hipblas_init<T>(hx_array[b], 1, N, incx);

            CHECK_HIP_ERROR(
                hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * sizeX, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(
            hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));

        /* =====================================================================
                    HIP BLAS
        =================================================================== */
        // device_pointer for d_rocblas_result
        {

            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

            status_1 = func(handle, N, dx_array, incx, batch_count, rocblas_result_device);

            CHECK_HIP_ERROR(hipMemcpy(h_rocblas_result_device,
                                      rocblas_result_device,
                                      sizeof(int) * batch_count,
                                      hipMemcpyDeviceToHost));
        }
        // host_pointer for rocblas_result2
        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

            status_2 = func(handle, N, dx_array, incx, batch_count, rocblas_result_host);
        }

        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_2 == HIPBLAS_STATUS_SUCCESS)
           && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            /* =====================================================================
                        CPU BLAS
            =================================================================== */
            for(int b = 0; b < batch_count; b++)
            {
                REFBLAS_FUNC(N, hx_array[b], incx, cpu_result + b);
                // change to Fortran 1 based indexing as in BLAS standard, not cblas zero based indexing
                cpu_result[b] += 1;
            }

            unit_check_general<int>(1, 1, batch_count, cpu_result, h_rocblas_result_device);
            unit_check_general<int>(1, 1, batch_count, cpu_result, rocblas_result_host);

        } // end of if unit/norm check
    }

    hipblasDestroy(handle);

    if(status_1 != HIPBLAS_STATUS_SUCCESS)
    {
        return status_1;
    }
    else if(status_2 != HIPBLAS_STATUS_SUCCESS)
    {
        return status_2;
    }
    else if(status_3 != HIPBLAS_STATUS_SUCCESS)
    {
        return status_3;
    }
    else
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
}

template <typename T>
hipblasStatus_t testing_amax_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasIamaxBatchedFn
        = FORTRAN ? hipblasIamaxBatched<T, true> : hipblasIamaxBatched<T, false>;

    return testing_iamax_iamin_batched<T, cblas_iamax<T>>(arg, hipblasIamaxBatchedFn);
}

template <typename T>
hipblasStatus_t testing_amin_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasIaminBatchedFn
        = FORTRAN ? hipblasIaminBatched<T, true> : hipblasIaminBatched<T, false>;

    return testing_iamax_iamin_batched<T, cblas_iamin<T>>(arg, hipblasIaminBatchedFn);
}
