/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "arg_check.h"
#include "testing_common.hpp"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_GemmBatched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemmBatchedFn
        = FORTRAN ? hipblasGemmBatched<T, true> : hipblasGemmBatched<T, false>;

    int M = argus.M;
    int N = argus.N;
    int K = argus.K;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    T h_alpha = argus.alpha;
    T h_beta  = argus.beta;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    int batch_count = argus.batch_count;

    if(batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0)
    {
        hipblasHandle_t handle;
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        hipblasCreate(&handle);

        const T *dA_array[1], *dB_array[1];
        T*       dC1_array[1];

        status = hipblasGemmBatchedFn(handle,
                                      transA,
                                      transB,
                                      M,
                                      N,
                                      K,
                                      &h_alpha,
                                      dA_array,
                                      lda,
                                      dB_array,
                                      ldb,
                                      &h_beta,
                                      dC1_array,
                                      ldc,
                                      batch_count);

        verify_hipblas_status_invalid_value(
            status,
            "ERROR: batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 ");

        hipblasDestroy(handle);

        return status;
    }

    T rocblas_error = 0.0;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;

    hipblasStatus_t status   = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;

    int A_row, A_col, B_row, B_col;

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M;
        A_col = K;
    }
    else
    {
        A_row = K;
        A_col = M;
    }

    if(transB == HIPBLAS_OP_N)
    {
        B_row = K;
        B_col = N;
    }
    else
    {
        B_row = N;
        B_col = K;
    }

    if(lda < A_row || ldb < B_row || ldc < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    int A_mat_size = A_col * lda;
    int B_mat_size = B_col * ldb;
    int C_mat_size = N * ldc;

    // arrays of pointers-to-host on host
    host_vector<T> hA_array[batch_count];
    host_vector<T> hB_array[batch_count];
    host_vector<T> hC_array[batch_count];
    host_vector<T> hC_copy_array[batch_count];

    // arrays of pointers-to-device on host
    device_batch_vector<T> dA_array(batch_count, A_mat_size);
    device_batch_vector<T> dB_array(batch_count, B_mat_size);
    device_batch_vector<T> dC1_array(batch_count, C_mat_size);
    device_batch_vector<T> dC2_array(batch_count, C_mat_size);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // arrays of pointers-to-device on device
    device_vector<T*, 0, T> dA_array_dev(batch_count);
    device_vector<T*, 0, T> dB_array_dev(batch_count);
    device_vector<T*, 0, T> dC1_array_dev(batch_count);
    device_vector<T*, 0, T> dC2_array_dev(batch_count);

    int last = batch_count - 1;
    if((!dA_array[last] && A_mat_size) || (!dB_array[last] && B_mat_size)
       || (!dC1_array[last] && C_mat_size) || (!dC2_array[last] && C_mat_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    hipError_t err_A, err_B, err_C_1, err_C_2, err_alpha, err_beta;
    srand(1);
    for(int i = 0; i < batch_count; i++)
    {
        hA_array[i]      = host_vector<T>(A_mat_size);
        hB_array[i]      = host_vector<T>(B_mat_size);
        hC_array[i]      = host_vector<T>(C_mat_size);
        hC_copy_array[i] = host_vector<T>(C_mat_size);

        // initialize matrices on host
        srand(1);
        hipblas_init<T>(hA_array[i], A_row, A_col, lda);
        hipblas_init<T>(hB_array[i], B_row, B_col, ldb);
        hipblas_init<T>(hC_array[i], M, N, ldc);

        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                hC_copy_array[i][i1 + i2 * ldc] = hC_array[i][i1 + i2 * ldc];
            }
        }

        // copy initialized matrices from host to device
        err_A = hipMemcpy(dA_array[i], hA_array[i], sizeof(T) * A_mat_size, hipMemcpyHostToDevice);
        err_B = hipMemcpy(dB_array[i], hB_array[i], sizeof(T) * B_mat_size, hipMemcpyHostToDevice);
        err_C_1
            = hipMemcpy(dC1_array[i], hC_array[i], sizeof(T) * C_mat_size, hipMemcpyHostToDevice);
        err_C_2
            = hipMemcpy(dC2_array[i], hC_array[i], sizeof(T) * C_mat_size, hipMemcpyHostToDevice);
        err_alpha = hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice);
        err_beta  = hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice);

        if((err_A != hipSuccess) || (err_C_1 != hipSuccess) || (err_alpha != hipSuccess)
           || (err_B != hipSuccess) || (err_C_2 != hipSuccess) || (err_beta != hipSuccess))
        {
            hipblasDestroy(handle);
            std::cerr << "dX_array[i] hipMemcpy error" << std::endl;
            return HIPBLAS_STATUS_MAPPING_ERROR;
        }
    }

    // copy array of pointers-to-device from host to device
    err_A
        = hipMemcpy(dA_array_dev, dA_array, batch_count * sizeof(*dA_array), hipMemcpyHostToDevice);
    err_B
        = hipMemcpy(dB_array_dev, dB_array, batch_count * sizeof(*dB_array), hipMemcpyHostToDevice);
    err_C_1 = hipMemcpy(
        dC1_array_dev, dC1_array, batch_count * sizeof(*dC1_array), hipMemcpyHostToDevice);
    err_C_2 = hipMemcpy(
        dC2_array_dev, dC2_array, batch_count * sizeof(*dC2_array), hipMemcpyHostToDevice);
    if((err_A != hipSuccess) || (err_B != hipSuccess) || (err_C_1 != hipSuccess)
       || (err_C_2 != hipSuccess))
    {
        hipblasDestroy(handle);
        std::cerr << "dX_array[i] hipMemcpy error" << std::endl;
        return HIPBLAS_STATUS_MAPPING_ERROR;
    }

    // calculate "golden" result on CPU
    for(int i = 0; i < batch_count; i++)
    {
        cblas_gemm<T>(transA,
                      transB,
                      M,
                      N,
                      K,
                      h_alpha,
                      (T*)hA_array[i],
                      lda,
                      (T*)hB_array[i],
                      ldb,
                      h_beta,
                      (T*)hC_copy_array[i],
                      ldc);
    }

    // test hipBLAS batched gemm with alpha and beta pointers on host
    {
        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = hipblasGemmBatchedFn(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        (const T* const*)dA_array_dev,
                                        lda,
                                        (const T* const*)dB_array_dev,
                                        ldb,
                                        &h_beta,
                                        dC2_array_dev,
                                        ldc,
                                        batch_count);

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            std::cout << "hipblasGemmBatched error" << std::endl;
            hipblasDestroy(handle);
            if(status_1 != HIPBLAS_STATUS_SUCCESS)
                return status_1;
            if(status_2 != HIPBLAS_STATUS_SUCCESS)
                return status_2;
        }

        for(int i = 0; i < batch_count; i++)
        {
            // copy result matrices from device to host
            err_C_2 = hipMemcpy(
                hC_array[i], dC2_array[i], sizeof(T) * C_mat_size, hipMemcpyDeviceToHost);

            if(err_C_2 != hipSuccess)
            {
                hipblasDestroy(handle);
                std::cerr << "dX_array[i] hipMemcpy error" << std::endl;
                return HIPBLAS_STATUS_MAPPING_ERROR;
            }

            // check hipBLAS result against "golden" result
            unit_check_general<T>(M, N, lda, hC_copy_array[i], hC_array[i]);
        }
    }

    // test hipBLAS batched gemm with alpha and beta pointers on device
    {
        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        status_2 = hipblasGemmBatchedFn(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        d_alpha,
                                        (const T* const*)dA_array_dev,
                                        lda,
                                        (const T* const*)dB_array_dev,
                                        ldb,
                                        d_beta,
                                        dC1_array_dev,
                                        ldc,
                                        batch_count);

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            std::cout << "hipblasGemmBatched error" << std::endl;
            hipblasDestroy(handle);
            if(status_1 != HIPBLAS_STATUS_SUCCESS)
                return status_1;
            if(status_2 != HIPBLAS_STATUS_SUCCESS)
                return status_2;
        }

        for(int i = 0; i < batch_count; i++)
        {
            // copy result matrices from device to host
            err_C_1 = hipMemcpy(
                hC_array[i], dC1_array[i], sizeof(T) * C_mat_size, hipMemcpyDeviceToHost);

            if(err_C_1 != hipSuccess)
            {
                hipblasDestroy(handle);
                std::cerr << "hC_array[i] hipMemcpy error" << std::endl;
                return HIPBLAS_STATUS_MAPPING_ERROR;
            }

            // check hipBLAS result against "golden" result
            unit_check_general<T>(M, N, lda, hC_copy_array[i], hC_array[i]);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
