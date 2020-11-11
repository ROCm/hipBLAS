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
hipblasStatus_t testing_axpy_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasAxpyStridedBatchedFn
        = FORTRAN ? hipblasAxpyStridedBatched<T, true> : hipblasAxpyStridedBatched<T, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;
    T      alpha        = argus.alpha;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    int stridex = N * abs_incx * stride_scale;
    int stridey = N * abs_incy * stride_scale;
    int sizeX   = stridex * batch_count;
    int sizeY   = stridey * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

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

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, abs_incx, stridex, batch_count);
    hipblas_init<T>(hy, 1, N, abs_incy, stridey, batch_count);

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasAxpyStridedBatchedFn(
        handle, N, &alpha, dx, incx, stridex, dy, incy, stridey, batch_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy<T>(
                N, alpha, hx_cpu.data() + b * stridex, incx, hy_cpu.data() + b * stridey, incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, stridex, hx_cpu.data(), hx.data());
            unit_check_general<T>(1, N, batch_count, abs_incy, stridey, hy_cpu.data(), hy.data());
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
