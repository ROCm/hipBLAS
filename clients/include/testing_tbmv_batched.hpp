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
hipblasStatus_t testing_tbmv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTbmvBatchedFn
        = FORTRAN ? hipblasTbmvBatched<T, true> : hipblasTbmvBatched<T, false>;

    int M    = argus.M;
    int K    = argus.K;
    int lda  = argus.lda;
    int incx = argus.incx;

    int A_size = lda * M;
    int X_size = M * incx;

    int batch_count = argus.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);
    hipblasStatus_t    status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || K < 0 || lda < M || incx == 0 || batch_count < 0)
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

    T alpha = (T)argus.alpha;
    T beta  = (T)argus.beta;

    // arrays of pointers-to-host on host
    host_vector<T> hA_array[batch_count];
    host_vector<T> hx_array[batch_count];
    host_vector<T> hres_array[batch_count];

    // arrays of pointers-to-device on host
    device_batch_vector<T> bA_array(batch_count, A_size);
    device_batch_vector<T> bx_array(batch_count, X_size);

    // arrays of pointers-to-device on device
    device_vector<T*, 0, T> dA_array(batch_count);
    device_vector<T*, 0, T> dx_array(batch_count);

    int last = batch_count - 1;
    if(!dA_array || !dx_array || (!bA_array[last] && A_size) || (!bx_array[last] && X_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    hipError_t err_A, err_x;
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA_array[b] = host_vector<T>(A_size);
        hx_array[b] = host_vector<T>(X_size);

        hres_array[b] = hx_array[b];

        // initialize matrices on host
        srand(1);
        hipblas_init<T>(hA_array[b], M, M, lda);
        hipblas_init<T>(hx_array[b], 1, M, incx);

        err_A = hipMemcpy(bA_array[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice);
        err_x = hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * X_size, hipMemcpyHostToDevice);

        if(err_A != hipSuccess || err_x != hipSuccess)
        {
            hipblasDestroy(handle);
            return HIPBLAS_STATUS_MAPPING_ERROR;
        }
    }

    err_A = hipMemcpy(dA_array, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_x = hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    if(err_A != hipSuccess || err_x != hipSuccess)
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_MAPPING_ERROR;
    }

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasTbmvBatchedFn(
            handle, uplo, transA, diag, M, K, dA_array, lda, dx_array, incx, batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hres_array[b], bx_array[b], sizeof(T) * X_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_tbmv<T>(uplo, transA, diag, M, K, hA_array[b], lda, hx_array[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, incx, hx_array, hres_array);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
