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
hipblasStatus_t testing_syr2_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSyr2StridedBatchedFn
        = FORTRAN ? hipblasSyr2StridedBatched<T, true> : hipblasSyr2StridedBatched<T, false>;

    int               N            = argus.N;
    int               incx         = argus.incx;
    int               incy         = argus.incy;
    int               lda          = argus.lda;
    char              char_uplo    = argus.uplo_option;
    hipblasFillMode_t uplo         = char2hipblas_fill(char_uplo);
    double            stride_scale = argus.stride_scale;
    int               batch_count  = argus.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    int strideA = lda * N * stride_scale;
    int stridex = abs_incx * N * stride_scale;
    int stridey = abs_incy * N * stride_scale;
    int A_size  = strideA * batch_count;
    int x_size  = stridex * batch_count;
    int y_size  = stridey * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || lda < 1 || lda < N || incx == 0 || incy == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_cpu(A_size);
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
    hipblas_init<T>(hA, N, N, lda, strideA, batch_count);
    hipblas_init<T>(hx, 1, N, abs_incx, stridex, batch_count);
    hipblas_init<T>(hy, 1, N, abs_incy, stridey, batch_count);
    hA_cpu = hA;

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
        status = hipblasSyr2StridedBatchedFn(handle,
                                             uplo,
                                             N,
                                             (T*)&alpha,
                                             dx,
                                             incx,
                                             stridex,
                                             dy,
                                             incy,
                                             stridey,
                                             dA,
                                             lda,
                                             strideA,
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
            cblas_syr2<T>(uplo,
                          N,
                          alpha,
                          hx.data() + b * stridex,
                          incx,
                          hy.data() + b * stridey,
                          incy,
                          hA_cpu.data() + b * strideA,
                          lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, lda, strideA, hA, hA_cpu);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
