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
hipblasStatus_t testing_trtri_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrtriStridedBatchedFn
        = FORTRAN ? hipblasTrtriStridedBatched<T, true> : hipblasTrtriStridedBatched<T, false>;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    int N           = argus.N;
    int lda         = argus.lda;
    int ldinvA      = lda;
    int batch_count = argus.batch_count;

    int strideA = lda * N;
    int A_size  = strideA * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(A_size);

    device_vector<T> dA(A_size);
    device_vector<T> dinvA(A_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;

    char char_uplo = argus.uplo_option;
    char char_diag = argus.diag_option;

    hipblasFillMode_t uplo = char2hipblas_fill(char_uplo);
    hipblasDiagType_t diag = char2hipblas_diagonal(char_diag);

    hipblasCreate(&handle);

    srand(1);
    hipblas_init_symmetric<T>(hA, N, lda, strideA, batch_count);
    for(int b = 0; b < batch_count; b++)
    {
        T* hAb = hA.data() + b * strideA;

        // proprocess the matrix to avoid ill-conditioned matrix
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hAb[i + j * lda] *= 0.01;

                if(j % 2)
                    hAb[i + j * lda] *= -1;
                if(uplo == HIPBLAS_FILL_MODE_LOWER && j > i)
                    hAb[i + j * lda] = 0.0f;
                else if(uplo == HIPBLAS_FILL_MODE_UPPER && j < i)
                    hAb[i + j * lda] = 0.0f;
                if(i == j)
                {
                    if(diag == HIPBLAS_DIAG_UNIT)
                        hAb[i + j * lda] = 1.0;
                    else
                        hAb[i + j * lda] *= 100.0;
                }
            }
        }
    }

    hB = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status = hipblasTrtriStridedBatchedFn(
        handle, uplo, diag, N, dA, lda, strideA, dinvA, ldinvA, strideA, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dinvA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            int info = cblas_trtri<T>(char_uplo, char_diag, N, hB.data() + b * strideA, lda);

            if(info != 0)
                printf("error in cblas_trtri\n");
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            for(int b = 0; b < batch_count; b++)
                near_check_general<T>(
                    N, N, lda, hB.data() + b * strideA, hA.data() + b * strideA, rel_error);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
