/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "arg_check.h"
#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

#define CLEANUP()                            \
    do                                       \
    {                                        \
        for(int i = 0; i < batch_count; i++) \
        {                                    \
            if(dA_array[i])                  \
                hipFree(dA_array[i]);        \
            if(dB_array[i])                  \
                hipFree(dB_array[i]);        \
            if(dC1_array[i])                 \
                hipFree(dC1_array[i]);       \
            if(dC2_array[i])                 \
                hipFree(dC2_array[i]);       \
                                             \
            if(hA_array[i])                  \
                free(hA_array[i]);           \
            if(hB_array[i])                  \
                free(hB_array[i]);           \
            if(hC_array[i])                  \
                free(hC_array[i]);           \
            if(hC_copy_array[i])             \
                free(hC_copy_array[i]);      \
        }                                    \
        if(hA_array)                         \
            free(hA_array);                  \
        if(hB_array)                         \
            free(hB_array);                  \
        if(hC_array)                         \
            free(hC_array);                  \
        if(hC_copy_array)                    \
            free(hC_copy_array);             \
                                             \
        if(dA_array)                         \
            free(dA_array);                  \
        if(dB_array)                         \
            free(dB_array);                  \
        if(dC1_array)                        \
            free(dC1_array);                 \
        if(dC2_array)                        \
            free(dC2_array);                 \
                                             \
        if(dA_array_dev)                     \
            hipFree(dA_array_dev);           \
        if(dB_array_dev)                     \
            hipFree(dB_array_dev);           \
        if(dC1_array_dev)                    \
            hipFree(dC1_array_dev);          \
    } while(0)

template <typename T>
hipblasStatus_t testing_GemmBatched(Arguments argus)
{
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

        status = hipblasGemmBatched<T>(handle,
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
    hipblasHandle_t handle;
    hipblasCreate(&handle);

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

    int A_mat_size = A_col * lda;
    int B_mat_size = B_col * ldb;
    int C_mat_size = N * ldc;

    // malloc arrays of pointers-to-host on host
    T** hA_array      = (T**)malloc(batch_count * sizeof(*hA_array));
    T** hB_array      = (T**)malloc(batch_count * sizeof(*hB_array));
    T** hC_array      = (T**)malloc(batch_count * sizeof(*hC_array));
    T** hC_copy_array = (T**)malloc(batch_count * sizeof(*hC_copy_array));

    // malloc arrays of pointers-to-device on host
    T** dA_array  = (T**)malloc(batch_count * sizeof(*dA_array));
    T** dB_array  = (T**)malloc(batch_count * sizeof(*dB_array));
    T** dC1_array = (T**)malloc(batch_count * sizeof(*dC1_array));
    T** dC2_array = (T**)malloc(batch_count * sizeof(*dC2_array));
    T * d_alpha, *d_beta;

    // Arrays of pointers-to-device on device
    T** dA_array_dev  = NULL;
    T** dB_array_dev  = NULL;
    T** dC1_array_dev = NULL;
    T** dC2_array_dev = NULL;

    if((!hA_array) || (!hB_array) || (!hC_array) || (!hC_copy_array) || (!dA_array) || (!dB_array)
       || (!dC1_array) || (!dC2_array))
    {
        CLEANUP();
        hipblasDestroy(handle);
        std::cerr << "malloc error" << std::endl;
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // malloc arrays of pointers-to-device on device
    hipError_t err_A, err_B, err_C_1, err_C_2, err_alpha, err_beta;
    err_A     = hipMalloc((void**)&dA_array_dev, batch_count * sizeof(*dA_array));
    err_B     = hipMalloc((void**)&dB_array_dev, batch_count * sizeof(*dB_array));
    err_C_1   = hipMalloc((void**)&dC1_array_dev, batch_count * sizeof(*dC1_array));
    err_C_2   = hipMalloc((void**)&dC2_array_dev, batch_count * sizeof(*dC2_array));
    err_alpha = hipMalloc(&d_alpha, sizeof(T));
    err_beta  = hipMalloc(&d_beta, sizeof(T));

    if((err_A != hipSuccess) || (err_C_1 != hipSuccess) || (err_alpha != hipSuccess)
       || (err_B != hipSuccess) || (err_C_2 != hipSuccess) || (err_beta != hipSuccess))
    {
        CLEANUP();
        hipblasDestroy(handle);
        std::cerr << "hipMalloc error" << std::endl;
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    srand(1);
    for(int i = 0; i < batch_count; i++)
    {
        // malloc matrices on host
        hA_array[i]      = (T*)malloc(A_mat_size * sizeof(hA_array[0][0]));
        hB_array[i]      = (T*)malloc(B_mat_size * sizeof(hB_array[0][0]));
        hC_array[i]      = (T*)malloc(C_mat_size * sizeof(hC_array[0][0]));
        hC_copy_array[i] = (T*)malloc(C_mat_size * sizeof(hC_copy_array[0][0]));

        if((!hA_array[i]) || (!hB_array[i]) || (!hC_array[i]) || (!hC_copy_array[i]))
        {
            CLEANUP();
            hipblasDestroy(handle);
            std::cerr << "hX_array[i] malloc error" << std::endl;
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }

        // malloc matrices on device
        err_A   = hipMalloc((void**)&dA_array[i], A_mat_size * sizeof(dA_array[0][0]));
        err_B   = hipMalloc((void**)&dB_array[i], B_mat_size * sizeof(dB_array[0][0]));
        err_C_1 = hipMalloc((void**)&dC1_array[i], C_mat_size * sizeof(dC1_array[0][0]));
        err_C_2 = hipMalloc((void**)&dC2_array[i], C_mat_size * sizeof(dC2_array[0][0]));

        if((err_A != hipSuccess) || (err_B != hipSuccess) || (err_C_1 != hipSuccess)
           || (err_C_2 != hipSuccess))
        {
            CLEANUP();
            hipblasDestroy(handle);
            std::cerr << "dX_array[i] hipMalloc error" << std::endl;
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }

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
            CLEANUP();
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
        CLEANUP();
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
                      hA_array[i],
                      lda,
                      hB_array[i],
                      ldb,
                      h_beta,
                      hC_copy_array[i],
                      ldc);
    }

    // test hipBLAS batched gemm with alpha and beta pointers on host
    {
        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = hipblasGemmBatched<T>(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         &h_alpha,
                                         (const T**)dA_array_dev,
                                         lda,
                                         (const T**)dB_array_dev,
                                         ldb,
                                         &h_beta,
                                         dC2_array_dev,
                                         ldc,
                                         batch_count);

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            std::cout << "hipblasGemmBatched error" << std::endl;
            CLEANUP();
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
                CLEANUP();
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

        status_2 = hipblasGemmBatched<T>(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         d_alpha,
                                         (const T**)dA_array_dev,
                                         lda,
                                         (const T**)dB_array_dev,
                                         ldb,
                                         d_beta,
                                         dC1_array_dev,
                                         ldc,
                                         batch_count);

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            std::cout << "hipblasGemmBatched error" << std::endl;
            CLEANUP();
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
                CLEANUP();
                hipblasDestroy(handle);
                std::cerr << "hC_array[i] hipMemcpy error" << std::endl;
                return HIPBLAS_STATUS_MAPPING_ERROR;
            }

            // check hipBLAS result against "golden" result
            unit_check_general<T>(M, N, lda, hC_copy_array[i], hC_array[i]);
        }
    }

    CLEANUP();
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
