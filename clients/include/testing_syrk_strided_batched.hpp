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
hipblasStatus_t testing_syrk_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSyrkStridedBatchedFn
        = FORTRAN ? hipblasSyrkStridedBatched<T, true> : hipblasSyrkStridedBatched<T, false>;

    int    N            = argus.N;
    int    K            = argus.K;
    int    lda          = argus.lda;
    int    ldc          = argus.ldc;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasFillMode_t  uplo     = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA   = char2hipblas_operation(argus.transA_option);
    int                K1       = (transA == HIPBLAS_OP_N ? K : N);
    int                stride_A = lda * K1 * stride_scale;
    int                stride_C = ldc * N * stride_scale;
    int                A_size   = stride_A * batch_count;
    int                C_size   = stride_C * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
       || (transA != HIPBLAS_OP_N && lda < K) || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hC(C_size);
    host_vector<T> hC2(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dC(C_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();
    T beta  = argus.get_beta<T>();

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, N, K1, lda, stride_A, batch_count);
    hipblas_init<T>(hC, N, N, ldc, stride_C, batch_count);

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasSyrkStridedBatchedFn(handle,
                                             uplo,
                                             transA,
                                             N,
                                             K,
                                             (T*)&alpha,
                                             dA,
                                             lda,
                                             stride_A,
                                             (T*)&beta,
                                             dC,
                                             ldc,
                                             stride_C,
                                             batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hC2.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_syrk<T>(uplo,
                          transA,
                          N,
                          K,
                          alpha,
                          hA.data() + b * stride_A,
                          lda,
                          beta,
                          hC.data() + b * stride_C,
                          ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, stride_C, hC2.data(), hC.data());
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
