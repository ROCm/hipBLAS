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
hipblasStatus_t testing_hbmv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHbmvBatchedFn
        = FORTRAN ? hipblasHbmvBatched<T, true> : hipblasHbmvBatched<T, false>;

    int N    = argus.N;
    int K    = argus.K;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    int A_size = lda * N;
    int X_size;
    int Y_size;

    int batch_count = argus.batch_count;

    hipblasFillMode_t uplo = char2hipblas_fill(argus.uplo_option);
    X_size                 = N * incx;
    Y_size                 = N * incy;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || lda < K + 1 || incx == 0 || incy == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();
    T beta  = argus.get_beta<T>();

    // arrays of pointers-to-host on host
    host_vector<T> hA_array[batch_count];
    host_vector<T> hx_array[batch_count];
    host_vector<T> hy_array[batch_count];
    host_vector<T> hz_array[batch_count];

    // arrays of pointers-to-device on host
    device_batch_vector<T> bA_array(batch_count, A_size);
    device_batch_vector<T> bx_array(batch_count, X_size);
    device_batch_vector<T> by_array(batch_count, Y_size);

    // arrays of pointers-to-device on device
    device_vector<T*, 0, T> dA_array(batch_count);
    device_vector<T*, 0, T> dx_array(batch_count);
    device_vector<T*, 0, T> dy_array(batch_count);

    int last = batch_count - 1;
    if(!dA_array || !dx_array || !dy_array || (!bA_array[last] && A_size)
       || (!bx_array[last] && X_size) || (!by_array[last] && Y_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    hipError_t err_A, err_x, err_y;
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA_array[b] = host_vector<T>(A_size);
        hx_array[b] = host_vector<T>(X_size);
        hy_array[b] = host_vector<T>(Y_size);
        hz_array[b] = host_vector<T>(Y_size);

        // initialize matrices on host
        srand(1);
        hipblas_init<T>(hA_array[b], K, N, lda);
        hipblas_init<T>(hx_array[b], 1, N, incx);
        hipblas_init<T>(hy_array[b], 1, N, incy);

        hz_array[b] = hy_array[b];
        err_A = hipMemcpy(bA_array[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice);
        err_x = hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * X_size, hipMemcpyHostToDevice);
        err_y = hipMemcpy(by_array[b], hy_array[b], sizeof(T) * Y_size, hipMemcpyHostToDevice);

        if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
        {
            hipblasDestroy(handle);
            return HIPBLAS_STATUS_MAPPING_ERROR;
        }
    }

    err_A = hipMemcpy(dA_array, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_x = hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_y = hipMemcpy(dy_array, by_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_MAPPING_ERROR;
    }

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasHbmvBatchedFn(handle,
                                      uplo,
                                      N,
                                      K,
                                      (T*)&alpha,
                                      dA_array,
                                      lda,
                                      dx_array,
                                      incx,
                                      (T*)&beta,
                                      dy_array,
                                      incy,
                                      batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            // here in cuda
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hy_array[b], by_array[b], sizeof(T) * Y_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_hbmv<T>(
                uplo, N, K, alpha, hA_array[b], lda, hx_array[b], incx, beta, hz_array[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, batch_count, incy, hz_array, hy_array);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
