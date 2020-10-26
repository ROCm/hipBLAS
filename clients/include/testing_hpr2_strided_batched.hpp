/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_hpr2_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHpr2StridedBatchedFn
        = FORTRAN ? hipblasHpr2StridedBatched<T, true> : hipblasHpr2StridedBatched<T, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int               dim_A    = N * (N + 1) / 2;
    int               stride_A = dim_A * stride_scale;
    int               stride_x = N * incx * stride_scale;
    int               stride_y = N * incy * stride_scale;
    int               A_size   = stride_A * batch_count;
    int               x_size   = stride_x * batch_count;
    int               y_size   = stride_y * batch_count;
    hipblasFillMode_t uplo     = char2hipblas_fill(argus.uplo_option);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx == 0 || incy == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hy(y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<T> dy(y_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, 1, dim_A, 1, stride_A, batch_count);
    hipblas_init<T>(hx, 1, N, incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, N, incy, stride_y, batch_count);

    // copy matrix is easy in STL; hB = hA: save a copy in hB which will be output of CPU BLAS
    hB = hA;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasHpr2StridedBatchedFn(handle,
                                             uplo,
                                             N,
                                             (T*)&alpha,
                                             dx,
                                             incx,
                                             stride_x,
                                             dy,
                                             incy,
                                             stride_y,
                                             dA,
                                             stride_A,
                                             batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hA.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_hpr2<T>(uplo,
                          N,
                          alpha,
                          hx.data() + b * stride_x,
                          incx,
                          hy.data() + b * stride_y,
                          incy,
                          hB.data() + b * stride_A);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, dim_A, batch_count, 1, stride_A, hB.data(), hA.data());
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
