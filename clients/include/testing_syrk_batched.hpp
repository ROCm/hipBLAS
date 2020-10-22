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
hipblasStatus_t testing_syrk_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSyrkBatchedFn
        = FORTRAN ? hipblasSyrkBatched<T, true> : hipblasSyrkBatched<T, false>;

    int N           = argus.N;
    int K           = argus.K;
    int lda         = argus.lda;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();
    T beta  = argus.get_beta<T>();

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

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    int K1     = (transA == HIPBLAS_OP_N ? K : N);
    int A_size = lda * K1;
    int C_size = ldc * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hC[batch_count];
    host_vector<T> hC2[batch_count];

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bC(batch_count, C_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dC(batch_count);

    int last = batch_count - 1;
    if(!dA || !dC || (!bA[last] && A_size) || (!bC[last] && C_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]  = host_vector<T>(A_size);
        hC[b]  = host_vector<T>(C_size);
        hC2[b] = host_vector<T>(C_size);

        srand(1);
        hipblas_init<T>(hA[b], N, K1, lda);
        hipblas_init<T>(hC[b], N, N, ldc);

        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bC[b], hC[b], sizeof(T) * C_size, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, bC, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasSyrkBatchedFn(
            handle, uplo, transA, N, K, (T*)&alpha, dA, lda, (T*)&beta, dC, ldc, batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hC2[b], bC[b], sizeof(T) * C_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_syrk<T>(uplo, transA, N, K, alpha, hA[b], lda, beta, hC[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, hC2, hC);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
