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
using hipblas_iamax_iamin_strided_batched_t = hipblasStatus_t (*)(
    hipblasHandle_t handle, int n, const T* x, int incx, int stridex, int batch_count, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
hipblasStatus_t testing_iamax_iamin_strided_batched(const Arguments&                         argus,
                                                    hipblas_iamax_iamin_strided_batched_t<T> func)
{
    int    N            = argus.N;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int stridex = N * incx * stride_scale;
    int sizeX   = stridex * batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // check to prevent undefined memory allocation error
    if(batch_count == 0)
    {
        // quick return success or invalid value
        device_vector<T>   dx(100);
        device_vector<int> d_rocblas_result(1);

        status_1 = func(handle, N, dx, incx, stridex, batch_count, d_rocblas_result);
    }
    else if(batch_count < 0)
    {
        status_1 = HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(N < 1 || incx <= 0)
    {
        // quick return success
        device_vector<T> dx(100);
        host_vector<int> h_rocblas_result(batch_count);
        host_vector<int> h_zeros(batch_count);
        for(int b = 0; b < batch_count; b++)
            h_zeros[b] = 0;

        status_1 = func(handle, N, dx, incx, stridex, batch_count, h_rocblas_result);
        unit_check_general<int>(1, 1, batch_count, h_zeros, h_rocblas_result);
    }
    else
    {
        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this
        // practice
        host_vector<T>     hx(sizeX);
        device_vector<T>   dx(sizeX);
        host_vector<int>   cpu_result(batch_count);
        host_vector<int>   rocblas_result1(batch_count);
        host_vector<int>   rocblas_result2(batch_count);
        device_vector<int> d_rocblas_result(batch_count);

        // Initial Data on CPU
        srand(1);
        hipblas_init<T>(hx, 1, N, incx, stridex, batch_count);

        // copy data from CPU to device, does not work for incx != 1
        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

        /* =====================================================================
                    HIP BLAS
        =================================================================== */
        // device_pointer for d_rocblas_result
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

            status_1 = func(handle, N, dx, incx, stridex, batch_count, d_rocblas_result);

            CHECK_HIP_ERROR(hipMemcpy(rocblas_result1,
                                      d_rocblas_result,
                                      sizeof(int) * batch_count,
                                      hipMemcpyDeviceToHost));
        }
        // host_pointer for rocblas_result2
        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

            status_2 = func(handle, N, dx, incx, stridex, batch_count, rocblas_result2.data());
        }

        if((status_1 == HIPBLAS_STATUS_SUCCESS) && (status_2 == HIPBLAS_STATUS_SUCCESS)
           && (status_3 == HIPBLAS_STATUS_SUCCESS))
        {
            /* =====================================================================
                        CPU BLAS
            =================================================================== */
            for(int b = 0; b < batch_count; b++)
            {
                REFBLAS_FUNC(N, hx.data() + b * stridex, incx, &(cpu_result[b]));
                // change to Fortran 1 based indexing as in BLAS standard, not cblas zero based indexing
                cpu_result[b] += 1;
            }

            unit_check_general<int>(1, 1, batch_count, cpu_result.data(), rocblas_result1.data());
            unit_check_general<int>(1, 1, batch_count, cpu_result.data(), rocblas_result2.data());

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
hipblasStatus_t testing_amax_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasIamaxStridedBatchedFn
        = FORTRAN ? hipblasIamaxStridedBatched<T, true> : hipblasIamaxStridedBatched<T, false>;

    return testing_iamax_iamin_strided_batched<T, cblas_iamax<T>>(arg,
                                                                  hipblasIamaxStridedBatchedFn);
}

template <typename T>
hipblasStatus_t testing_amin_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasIaminStridedBatchedFn
        = FORTRAN ? hipblasIaminStridedBatched<T, true> : hipblasIaminStridedBatched<T, false>;

    return testing_iamax_iamin_strided_batched<T, cblas_iamin<T>>(arg,
                                                                  hipblasIaminStridedBatchedFn);
}
