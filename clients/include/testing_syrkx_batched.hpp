/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
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
hipblasStatus_t testing_syrkx_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSyrkxBatchedFn
        = FORTRAN ? hipblasSyrkxBatched<T, true> : hipblasSyrkxBatched<T, false>;

    int N           = argus.N;
    int K           = argus.K;
    int lda         = argus.lda;
    int ldb         = argus.ldb;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    T alpha = argus.get_alpha<T>();
    T beta  = argus.get_beta<T>();

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && (lda < N || ldb < N))
       || (transA != HIPBLAS_OP_N && (lda < K || ldb < K)) || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int K1     = (transA == HIPBLAS_OP_N ? K : N);
    int A_size = lda * K1;
    int B_size = ldb * K1;
    int C_size = ldc * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_array[batch_count];
    host_vector<T> hB_array[batch_count];
    host_vector<T> hC_array[batch_count];
    host_vector<T> hC_array_copy[batch_count];

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bB(batch_count, B_size);
    device_batch_vector<T> bC(batch_count, C_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);
    device_vector<T*, 0, T> dC(batch_count);

    int last = batch_count - 1;
    if(!dA || !dB || !dC || (!bA[last] && A_size) || (!bB[last] && B_size) || (!bC[last] && C_size))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used = 0.0;
    double hipblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA_array[b]      = host_vector<T>(A_size);
        hB_array[b]      = host_vector<T>(B_size);
        hC_array[b]      = host_vector<T>(C_size);
        hC_array_copy[b] = host_vector<T>(C_size);

        srand(1);
        hipblas_init<T>(hA_array[b], N, K1, lda);
        hipblas_init<T>(hB_array[b], N, K1, ldb);
        hipblas_init<T>(hC_array[b], N, N, ldc);

        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bB[b], hB_array[b], sizeof(T) * B_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bC[b], hC_array[b], sizeof(T) * C_size, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, bB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, bC, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    status = hipblasSyrkxBatchedFn(
        handle, uplo, transA, N, K, (T*)&alpha, dA, lda, dB, ldb, (T*)&beta, dC, ldc, batch_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hC_array_copy[b], bC[b], sizeof(T) * C_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            syrkx_reference<T>(uplo,
                               transA,
                               N,
                               K,
                               alpha,
                               hA_array[b],
                               lda,
                               hB_array[b],
                               ldb,
                               beta,
                               hC_array[b],
                               ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, hC_array_copy, hC_array);
        }
        if(argus.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', N, N, ldc, hC_array_copy, hC_array, batch_count);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        status = hipblasGetStream(handle, &stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            status = hipblasSyrkxBatchedFn(handle,
                                           uplo,
                                           transA,
                                           N,
                                           K,
                                           (T*)&alpha,
                                           dA,
                                           lda,
                                           dB,
                                           ldb,
                                           (T*)&beta,
                                           dC,
                                           ldc,
                                           batch_count);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_N,
                      e_K,
                      e_lda,
                      e_ldb,
                      e_ldc,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         syrkx_gflop_count<T>(N, K),
                         syrkx_gbyte_count<T>(N, K),
                         hipblas_error);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
