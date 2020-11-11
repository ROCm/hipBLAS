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
hipblasStatus_t testing_trmm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmmStridedBatchedFn
        = FORTRAN ? hipblasTrmmStridedBatched<T, true> : hipblasTrmmStridedBatched<T, false>;

    int    M            = argus.M;
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasSideMode_t  side   = char2hipblas_side(argus.side_option);
    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int K        = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int stride_A = lda * K * stride_scale;
    int stride_B = ldb * N * stride_scale;

    int A_size = stride_A * batch_count;
    int B_size = stride_B * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hB_copy(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();
    T beta  = argus.get_beta<T>();

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init_symmetric<T>(hA, K, lda, stride_A, batch_count);
    hipblas_init<T>(hB, M, N, ldb, stride_B, batch_count);
    hB_copy = hB;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasTrmmStridedBatchedFn(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             M,
                                             N,
                                             &alpha,
                                             dA,
                                             lda,
                                             stride_A,
                                             dB,
                                             ldb,
                                             stride_B,
                                             batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            // here in cuda
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hB_copy.data(), dB, sizeof(T) * B_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          alpha,
                          hA.data() + b * stride_A,
                          lda,
                          hB.data() + b * stride_B,
                          ldb);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldb, stride_B, hB, hB_copy);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
