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

template <typename T, typename U>
hipblasStatus_t testing_herkx_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHerkxBatchedFn
        = FORTRAN ? hipblasHerkxBatched<T, U, true> : hipblasHerkxBatched<T, U, false>;

    int N           = argus.N;
    int K           = argus.K;
    int lda         = argus.lda;
    int ldb         = argus.ldb;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();
    U beta  = argus.get_beta<U>();

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

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    int K1     = (transA == HIPBLAS_OP_N ? K : N);
    int A_size = lda * K1;
    int B_size = ldb * K1;
    int C_size = ldc * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hB[batch_count];
    host_vector<T> hC[batch_count];
    host_vector<T> hC2[batch_count];

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bB(batch_count, B_size);
    device_batch_vector<T> bC(batch_count, C_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);
    device_vector<T*, 0, T> dC(batch_count);

    int last = batch_count - 1;
    if(!dA || !dB || !dC || (!bA[last] && A_size) || (!bB[last] && B_size) || (!bC[last] && C_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]  = host_vector<T>(A_size);
        hB[b]  = host_vector<T>(B_size);
        hC[b]  = host_vector<T>(C_size);
        hC2[b] = host_vector<T>(C_size);

        srand(1);
        hipblas_init<T>(hA[b], N, K1, lda);
        hipblas_init<T>(hB[b], N, K1, ldb);
        hipblas_init<T>(hC[b], N, N, ldc);

        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bB[b], hB[b], sizeof(T) * B_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bC[b], hC[b], sizeof(T) * C_size, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, bB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
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
        status = hipblasHerkxBatchedFn(handle,
                                       uplo,
                                       transA,
                                       N,
                                       K,
                                       (T*)&alpha,
                                       dA,
                                       lda,
                                       dB,
                                       ldb,
                                       (U*)&beta,
                                       dC,
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
        hipMemcpy(hC2[b], bC[b], sizeof(T) * C_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_herkx<T>(uplo, transA, N, K, alpha, hA[b], lda, hB[b], ldb, beta, hC[b], ldc);
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
