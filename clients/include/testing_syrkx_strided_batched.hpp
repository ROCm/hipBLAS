/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_syrkx_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSyrkxStridedBatchedFn
        = FORTRAN ? hipblasSyrkxStridedBatched<T, true> : hipblasSyrkxStridedBatched<T, false>;

    int    N            = argus.N;
    int    K            = argus.K;
    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    int    ldc          = argus.ldc;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

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

    int           K1       = transA == HIPBLAS_OP_N ? K : N;
    hipblasStride stride_A = size_t(lda) * K1 * stride_scale;
    hipblasStride stride_B = size_t(ldb) * K1 * stride_scale;
    hipblasStride stride_C = size_t(ldc) * N * stride_scale;
    int           A_size   = stride_A * batch_count;
    int           B_size   = stride_B * batch_count;
    int           C_size   = stride_C * batch_count;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC(C_size);
    host_vector<T> hC_copy(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);

    double gpu_time_used = 0.0;
    double hipblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, N, K1, lda, stride_A, batch_count);
    hipblas_init<T>(hB, N, K1, ldb, stride_B, batch_count);
    hipblas_init<T>(hC, N, N, ldc, stride_C, batch_count);

    hC_copy = hC;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice);

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    status = hipblasSyrkxStridedBatchedFn(handle,
                                          uplo,
                                          transA,
                                          N,
                                          K,
                                          (T*)&alpha,
                                          dA,
                                          lda,
                                          stride_A,
                                          dB,
                                          ldb,
                                          stride_B,
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

    // copy output from device to CPU
    hipMemcpy(hC.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost);

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int batch = 0; batch < batch_count; batch++)
        {
            // B must == A to use syrk as reference
            syrkx_reference<T>(uplo,
                               transA,
                               N,
                               K,
                               alpha,
                               hA.data() + batch * stride_A,
                               lda,
                               hB.data() + batch * stride_B,
                               ldb,
                               beta,
                               hC_copy.data() + batch * stride_C,
                               ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, stride_C, hC_copy, hC);
        }
        if(argus.norm_check)
        {
            hipblas_error = std::abs(norm_check_general<T>(
                'F', N, N, ldc, stride_C, hC_copy.data(), hC.data(), batch_count));
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

            status = hipblasSyrkxStridedBatchedFn(handle,
                                                  uplo,
                                                  transA,
                                                  N,
                                                  K,
                                                  (T*)&alpha,
                                                  dA,
                                                  lda,
                                                  stride_A,
                                                  dB,
                                                  ldb,
                                                  stride_B,
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
