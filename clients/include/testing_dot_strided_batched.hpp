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

template <typename T, bool CONJ = false>
hipblasStatus_t testing_dot_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDotStridedBatchedFn
        = FORTRAN
              ? (CONJ ? hipblasDotcStridedBatched<T, true> : hipblasDotStridedBatched<T, true>)
              : (CONJ ? hipblasDotcStridedBatched<T, false> : hipblasDotStridedBatched<T, false>);

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int stridex = N * incx * stride_scale;
    int stridey = N * incy * stride_scale;
    int sizeX   = stridex * batch_count;
    int sizeY   = stridey * batch_count;

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
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> h_rocblas_result1(batch_count);
    host_vector<T> h_rocblas_result2(batch_count);
    host_vector<T> h_cpu_result(batch_count);

    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);
    device_vector<T> d_rocblas_result(batch_count);

    int device_pointer = 1;

    // TODO: Change to 1 when rocBLAS is fixed.
    int host_pointer = 0;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init_alternating_sign<T>(hx, 1, N, incx, stridex, batch_count);
    hipblas_init<T>(hy, 1, N, incy, stridey, batch_count);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    // hipblasDot accept both dev/host pointer for the scalar
    if(device_pointer)
    {

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        status_2 = (hipblasDotStridedBatchedFn)(
            handle, N, dx, incx, stridex, dy, incy, stridey, batch_count, d_rocblas_result);
    }
    if(host_pointer)
    {

        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_4 = (hipblasDotStridedBatchedFn)(
            handle, N, dx, incx, stridex, dy, incy, stridey, batch_count, h_rocblas_result2);
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
            h_rocblas_result1, d_rocblas_result, sizeof(T) * batch_count, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<T> : cblas_dot<T>)(N,
                                                  hx.data() + b * stridex,
                                                  incx,
                                                  hy.data() + b * stridey,
                                                  incy,
                                                  &h_cpu_result[b]);
        }

        if(argus.unit_check)
        {
            unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_rocblas_result1);
            // unit_check_general<T>(1, batch_count, 1, h_cpu_result, h_rocblas_result2);
        }

    } // end of if unit/norm check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_dotc_strided_batched(const Arguments& argus)
{
    return testing_dot_strided_batched<T, true>(argus);
}
