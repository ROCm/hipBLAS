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
hipblasStatus_t testing_trtri_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrtriBatchedFn
        = FORTRAN ? hipblasTrtriBatched<T, true> : hipblasTrtriBatched<T, false>;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    int N = argus.N;
    int lda;
    int ldinvA;
    ldinvA = lda    = argus.lda;
    int batch_count = argus.batch_count;

    int A_size = size_t(lda) * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hB[batch_count];

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bB(batch_count, A_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dinvA(batch_count);

    int last = batch_count - 1;
    if(!dA || !dinvA || !bA[last] || !bB[last])
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;

    char char_uplo = argus.uplo_option;
    char char_diag = argus.diag_option;

    hipblasFillMode_t uplo = char2hipblas_fill(char_uplo);
    hipblasDiagType_t diag = char2hipblas_diagonal(char_diag);

    hipblasCreate(&handle);

    for(int b = 0; b < batch_count; b++)
    {
        hA[b] = host_vector<T>(A_size);
        hB[b] = host_vector<T>(A_size);

        srand(1);
        hipblas_init_symmetric<T>(hA[b], N, lda);

        // proprocess the matrix to avoid ill-conditioned matrix
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * lda] *= 0.01;

                if(j % 2)
                    hA[b][i + j * lda] *= -1;
                if(uplo == HIPBLAS_FILL_MODE_LOWER && j > i)
                    hA[b][i + j * lda] = 0.0f;
                else if(uplo == HIPBLAS_FILL_MODE_UPPER && j < i)
                    hA[b][i + j * lda] = 0.0f;
                if(i == j)
                {
                    if(diag == HIPBLAS_DIAG_UNIT)
                        hA[b][i + j * lda] = 1.0;
                    else
                        hA[b][i + j * lda] *= 100.0;
                }
            }
        }

        hB[b] = hA[b];

        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bB[b], hB[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, bB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status = hipblasTrtriBatchedFn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
        CHECK_HIP_ERROR(hipMemcpy(hA[b], bB[b], sizeof(T) * A_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            int info = cblas_trtri<T>(char_uplo, char_diag, N, hB[b].data(), lda);

            if(info != 0)
                printf("error in cblas_trtri\n");
        }

#ifndef NDEBUG
        //print_matrix(hB, hA, N, N, lda);
#endif

        if(argus.unit_check)
        {
            for(int b = 0; b < batch_count; b++)
                near_check_general<T>(N, N, lda, hB[b].data(), hA[b].data(), rel_error);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
