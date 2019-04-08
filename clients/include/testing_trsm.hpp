/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trsm(Arguments argus)
{

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char char_side   = argus.side_option;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;
    T    alpha       = argus.alpha;

    hipblasSideMode_t  side   = char2hipblas_side(char_side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(char_diag);

    int K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int A_size = lda * K;
    int B_size = ldb * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hB_copy(B_size);
    vector<T> hX(B_size);

    T *dA, *dB;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;

    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));

    // Initial hA on CPU
    srand(1);
    hipblas_init_symmetric<T>(hA, K, lda);
    // pad untouched area into zero
    for(int i = K; i < lda; i++)
    {
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = 0.0;
        }
    }
    // proprocess the matrix to avoid ill-conditioned matrix
    vector<int> ipiv(K);
    cblas_getrf(K, K, hA.data(), lda, ipiv.data());
    for(int i = 0; i < K; i++)
    {
        for(int j = i; j < K; j++)
        {
            hA[i + j * lda] = hA[j + i * lda];
            if(diag == HIPBLAS_DIAG_UNIT)
            {
                if(i == j)
                    hA[i + j * lda] = 1.0;
            }
        }
    }

    // Initial hB, hX on CPU
    hipblas_init<T>(hB, M, N, ldb);
    // pad untouched area into zero
    for(int i = M; i < ldb; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hB[i + j * ldb] = 0.0;
        }
    }
    hX = hB; // original solution hX

    // Calculate hB = hA*hX;
    cblas_trmm<T>(
        side, uplo, transA, diag, M, N, 1.0 / alpha, (const T*)hA.data(), lda, hB.data(), ldb);

    hB_copy = hB;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasTrsm<T>(handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb);

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB.data(), dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_trsm<T>(
            side, uplo, transA, diag, M, N, alpha, (const T*)hA.data(), lda, hB_copy.data(), ldb);

        //      print_matrix(hB_copy, hB, min(M, 3), min(N,3), ldb);

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        rocblas_error = norm_check_general<T>('F', M, N, ldb, hB_copy.data(), hB.data());
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
