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
hipblasStatus_t testing_trmm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmmBatchedFn
        = FORTRAN ? hipblasTrmmBatched<T, true> : hipblasTrmmBatched<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    hipblasSideMode_t  side   = char2hipblas_side(argus.side_option);
    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);
    hipblasStatus_t    status = HIPBLAS_STATUS_SUCCESS;

    int K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int A_size = lda * K;
    int B_size = ldb * N;

    int batch_count = argus.batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    double gpu_time_used = 0.0;
    double hipblas_error = 0.0;

    T alpha = argus.get_alpha<T>();
    T beta  = argus.get_beta<T>();

    // arrays of pointers-to-host on host
    host_vector<T> hA_array[batch_count];
    host_vector<T> hB_array[batch_count];
    host_vector<T> hB_copy_array[batch_count];

    // arrays of pointers-to-device on host
    device_batch_vector<T>  bA_array(batch_count, A_size);
    device_batch_vector<T>  bB_array(batch_count, B_size);
    device_vector<T*, 0, T> dA_array(batch_count);
    device_vector<T*, 0, T> dB_array(batch_count);

    int last = batch_count - 1;
    if(!dA_array || !dB_array || (!bA_array[last] && A_size) || (!bB_array[last] && B_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    hipError_t err_A, err_B;
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA_array[b] = host_vector<T>(A_size);
        hB_array[b] = host_vector<T>(B_size);

        hB_copy_array[b] = hB_array[b];

        // initialize matrices on host
        srand(1);
        hipblas_init_symmetric<T>(hA_array[b], K, lda);
        hipblas_init<T>(hB_array[b], M, N, ldb);
        hB_copy_array[b] = hB_array[b];

        CHECK_HIP_ERROR(
            hipMemcpy(bA_array[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(bB_array[b], hB_array[b], sizeof(T) * B_size, hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dA_array, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB_array, bB_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    status = hipblasTrmmBatchedFn(
        handle, side, uplo, transA, diag, M, N, &alpha, dA_array, lda, dB_array, ldb, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hB_copy_array[b], bB_array[b], sizeof(T) * B_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmm<T>(
                side, uplo, transA, diag, M, N, alpha, hA_array[b], lda, hB_array[b], ldb);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldb, hB_array, hB_copy_array);
        }
        if(argus.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', M, N, ldb, hB_copy_array, hB_array, batch_count);
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

            status = hipblasTrmmBatchedFn(handle,
                                          side,
                                          uplo,
                                          transA,
                                          diag,
                                          M,
                                          N,
                                          &alpha,
                                          dA_array,
                                          lda,
                                          dB_array,
                                          ldb,
                                          batch_count);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_N,
                      e_lda,
                      e_ldb,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, K),
                         trmm_gbyte_count<T>(M, N, K),
                         hipblas_error);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
