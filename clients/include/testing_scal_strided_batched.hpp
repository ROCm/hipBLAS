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
hipblasStatus_t testing_scal_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasScalStridedBatchedFn
        = FORTRAN ? hipblasScalStridedBatched<T, U, true> : hipblasScalStridedBatched<T, U, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int stridex = N * incx * stride_scale;
    int sizeX   = stridex * batch_count;

    U alpha = argus.alpha;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hz(sizeX);

    device_vector<T> dx(sizeX);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx, stridex, batch_count);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasScalStridedBatchedFn(handle, N, &alpha, dx, incx, stridex, batch_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<T, U>(N, alpha, hz.data() + b * stridex, incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, stridex, hz, hx);
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
