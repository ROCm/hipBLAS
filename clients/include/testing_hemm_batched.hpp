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
hipblasStatus_t testing_hemm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHemmBatchedFn
        = FORTRAN ? hipblasHemmBatched<T, true> : hipblasHemmBatched<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    hipblasSideMode_t side   = char2hipblas_side(argus.side_option);
    hipblasFillMode_t uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasStatus_t   status = HIPBLAS_STATUS_SUCCESS;

    int K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int A_size = lda * K;
    int B_size = ldb * N;
    int C_size = ldc * N;

    int batch_count = argus.batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || ldc < M || batch_count < 0)
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
    host_vector<T> hB_array[batch_count];
    host_vector<T> hC_array[batch_count];
    host_vector<T> hC_gold_array[batch_count];

    // arrays of pointers-to-device on host
    device_batch_vector<T> bA_array(batch_count, A_size);
    device_batch_vector<T> bB_array(batch_count, B_size);
    device_batch_vector<T> bC_array(batch_count, C_size);

    // arrays of pointers-to-device on device
    device_vector<T*, 0, T> dA_array(batch_count);
    device_vector<T*, 0, T> dB_array(batch_count);
    device_vector<T*, 0, T> dC_array(batch_count);

    int last = batch_count - 1;
    if(!dA_array || !dB_array || !dC_array || (!bA_array[last] && A_size)
       || (!bB_array[last] && B_size) || (!bC_array[last] && C_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    hipError_t err_A, err_B, err_C;
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA_array[b]      = host_vector<T>(A_size);
        hB_array[b]      = host_vector<T>(B_size);
        hC_array[b]      = host_vector<T>(C_size);
        hC_gold_array[b] = host_vector<T>(C_size);

        // initialize matrices on host
        srand(1);
        hipblas_init<T>(hA_array[b], M, N, lda);
        hipblas_init<T>(hB_array[b], M, N, ldb);
        hipblas_init<T>(hC_array[b], M, N, ldc);
        hC_gold_array[b] = hC_array[b];

        err_A = hipMemcpy(bA_array[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice);
        err_B = hipMemcpy(bB_array[b], hB_array[b], sizeof(T) * B_size, hipMemcpyHostToDevice);
        err_C = hipMemcpy(bC_array[b], hC_array[b], sizeof(T) * C_size, hipMemcpyHostToDevice);

        if(err_A != hipSuccess || err_B != hipSuccess || err_C != hipSuccess)
        {
            hipblasDestroy(handle);
            return HIPBLAS_STATUS_MAPPING_ERROR;
        }
    }

    err_A = hipMemcpy(dA_array, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_B = hipMemcpy(dB_array, bB_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_C = hipMemcpy(dC_array, bC_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    if(err_A != hipSuccess || err_B != hipSuccess || err_C != hipSuccess)
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_MAPPING_ERROR;
    }

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasHemmBatchedFn(handle,
                                      side,
                                      uplo,
                                      M,
                                      N,
                                      &alpha,
                                      dA_array,
                                      lda,
                                      dB_array,
                                      ldb,
                                      &beta,
                                      dC_array,
                                      ldc,
                                      batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hC_array[b], bC_array[b], sizeof(T) * C_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_hemm<T>(side,
                          uplo,
                          M,
                          N,
                          alpha,
                          hA_array[b],
                          lda,
                          hB_array[b],
                          ldb,
                          beta,
                          hC_gold_array[b],
                          ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_gold_array, hC_array);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
