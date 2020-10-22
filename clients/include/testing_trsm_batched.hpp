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
hipblasStatus_t testing_trsm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrsmBatchedFn
        = FORTRAN ? hipblasTrsmBatched<T, true> : hipblasTrsmBatched<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char char_side   = argus.side_option;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;
    T    alpha       = argus.alpha;
    int  batch_count = argus.batch_count;

    hipblasSideMode_t  side   = char2hipblas_side(char_side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(char_diag);

    int K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int A_size = lda * K;
    int B_size = ldb * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    // TODO: Workaround for cuda tests, not actually testing return values
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!M || !N || !lda || !ldb || !batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hB[batch_count];
    host_vector<T> hB_copy[batch_count];
    host_vector<T> hX[batch_count];

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bB(batch_count, B_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);

    int last = batch_count - 1;
    if(!dA || !dB || (!bA[last] && A_size) || (!bB[last] && B_size))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]      = host_vector<T>(A_size);
        hB[b]      = host_vector<T>(B_size);
        hB_copy[b] = host_vector<T>(B_size);
        hX[b]      = host_vector<T>(B_size);

        hipblas_init_symmetric<T>(hA[b], K, lda);
        // pad untouched area into zero
        for(int i = K; i < lda; i++)
        {
            for(int j = 0; j < K; j++)
            {
                hA[b][i + j * lda] = 0.0;
            }
        }

        // proprocess the matrix to avoid ill-conditioned matrix
        vector<int> ipiv(K);
        cblas_getrf(K, K, hA[b].data(), lda, ipiv.data());
        for(int i = 0; i < K; i++)
        {
            for(int j = i; j < K; j++)
            {
                hA[b][i + j * lda] = hA[b][j + i * lda];
                if(diag == HIPBLAS_DIAG_UNIT)
                {
                    if(i == j)
                        hA[b][i + j * lda] = 1.0;
                }
            }
        }

        // Initial hB, hX on CPU
        hipblas_init<T>(hB[b], M, N, ldb);
        // pad untouched area into zero
        for(int i = M; i < ldb; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hB[b][i + j * ldb] = 0.0;
            }
        }
        hX[b] = hB[b]; // original solution hX

        // Calculate hB = hA*hX;
        cblas_trmm<T>(side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      T(1.0) / alpha,
                      (const T*)hA[b].data(),
                      lda,
                      hB[b].data(),
                      ldb);

        hB_copy[b] = hB[b];

        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bB[b], hB[b], sizeof(T) * B_size, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, bB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasTrsmBatchedFn(
        handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, batch_count);

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
        CHECK_HIP_ERROR(hipMemcpy(hB[b], bB[b], sizeof(T) * B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trsm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          alpha,
                          (const T*)hA[b].data(),
                          lda,
                          hB_copy[b].data(),
                          ldb);
        }

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        for(int b = 0; b < batch_count; b++)
        {
            double error = norm_check_general<T>('F', M, N, ldb, hB_copy[b].data(), hB[b].data());
            unit_check_error(error, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
